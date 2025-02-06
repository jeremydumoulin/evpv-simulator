# coding: utf-8

import json
import os
import rasterio
import pandas as pd
import geopandas as gpd
import random
import numpy as np
from shapely.geometry import shape, LineString, Point, Polygon, box, MultiPoint
from shapely.ops import transform, nearest_points, snap
from geopy.distance import geodesic, distance
import openrouteservice
import time
import math
import csv
import folium
import branca.colormap as cm

from evpv.vehicle import Vehicle
from evpv.vehiclefleet import VehicleFleet
from evpv.region import Region
from evpv.mobilitysimulator import MobilitySimulator
from evpv.pvsimulator import PVSimulator
from evpv import helpers as hlp

class ChargingSimulator:
    """
    A class to simulate the spatio-temporal daily charging demand of an electric vehicle fleet within a specified region.

    This class models charging demand by integrating data from mobility simulations, region-specific characteristics, 
    and vehicle fleet details, along with user-defined charging scenarios that include charging efficiency and behavior settings.

    Key Features:

    - Spatial Charging Demand: Computes the charging demand per traffic zone within the specified region, enabling geographically detailed assessments of charging needs.
    - Temporal Charging Demand: Generates a load profile for each electric vehicle across all traffic zones using stochastic allocation.
      The number of vehicles charging and charging times are based on a state-of-charge (SOC) decision model, with a "dumb charging" 
      strategy by default (where vehicles charge at full charger power upon arrival).
    - Scenario-Based Modeling: Allows flexible scenario configuration, including charging locations (with varying charger power levels), 
      and custom arrival times at home and work, to reflect realistic charging patterns.
    - Smart Charging Compatibility: Enables the application of smart charging strategies in a secondary processing step, with 
      pre-implemented strategies such as peak shaving to manage grid impact.

    Note: This class assumes the use of a predefined region, vehicle fleet, and mobility demand simulator. Charging scenario parameters are provided
    as a dictionary, allowing flexible configuration of different charging behaviors.
    """

    def __init__(self, region: Region, vehicle_fleet: VehicleFleet, mobility_demand: MobilitySimulator, scenario: dict, charging_efficiency: float):
        """
        Initializes the ChargingSimulator class.

        Args:
            region (Region): An instance representing the geographic area for the simulation.
            vehicle_fleet (VehicleFleet): The electric vehicle fleet to be simulated.
            mobility_demand (MobilitySimulator): An instance of MobilitySimulator providing mobility demand data.
            scenario (dict): Configuration parameters for the charging scenario. Keys:
                'charging_power' (float): The average power used for charging (in kW).
                'max_charging_sessions' (int): Maximum number of charging sessions per day.
                'charging_schedule' (str): Type of schedule for charging (e.g., 'daytime', 'nighttime').
            charging_efficiency (float): The efficiency factor for charging, ranging between 0 and 1.

        Prints:
            Initialization details, including the chosen region, vehicle fleet characteristics, and scenario configuration.
        """
        print("=========================================")
        print(f"INFO \t Creation of a ChargingSimulator object.")
        print("=========================================")

        self.vehicle_fleet = vehicle_fleet
        self.region = region
        self.mobility_demand = mobility_demand   
        self.scenario = scenario
        self.charging_efficiency = charging_efficiency       

        print(f"INFO \t Successful initialization of input parameters.")

        # Modeling results
        self._spatial_demand = None
        self._temporal_demand_vehicle_properties = None
        self._temporal_demand_profile = None
        self._temporal_demand_profile_aggregated = None

    # Results properties (read-only)

    @property
    def spatial_demand(self) -> pd.DataFrame:
        """pd.DataFrame: The spatial charging demand."""
        return self._spatial_demand

    @property
    def temporal_demand_vehicle_properties(self) -> pd.DataFrame:
        """pd.DataFrame: The properties of the vehicles that are charging today. """
        return self._temporal_demand_vehicle_properties

    @property
    def temporal_demand_profile(self) -> pd.DataFrame:
        """pd.DataFrame: The temporal charging demand for each vehicle."""
        return self._temporal_demand_profile

    @property
    def temporal_demand_profile_aggregated(self) -> pd.DataFrame:
        """pd.DataFrame: The total temporal charging demand (aggregated over each vehicle)."""
        return self._temporal_demand_profile_aggregated

    # Properties and Setters 

    @property
    def region(self) -> Region:
        """Region: The region used in the simulation."""
        return self._region

    @region.setter
    def region(self, value: Region):
        if not isinstance(value, Region):
            raise ValueError("region must be an instance of the Region class.")
        self._region = value

    @property
    def vehicle_fleet(self) -> VehicleFleet:
        """VehicleFleet: The vehicle fleet used in the simulation."""
        return self._vehicle_fleet

    @vehicle_fleet.setter
    def vehicle_fleet(self, value: VehicleFleet):
        if not isinstance(value, VehicleFleet):
            raise ValueError("vehicle_fleet must be an instance of the VehicleFleet class.")
        self._vehicle_fleet = value

    @property
    def mobility_demand(self) -> MobilitySimulator:
        """MobilitySimulator: The mobility demand (i.e. flows between traffic zones) aggregated over all simulations provided."""
        return self._mobility_demand

    @mobility_demand.setter
    def mobility_demand(self, value: MobilitySimulator):
        if not isinstance(value, MobilitySimulator):
            raise ValueError(f"mobility_demand object must be an instance of the MobilitySimulator class.")
        if not 'n_outflows' in value.aggregated_zone_metrics.columns:
            raise ValueError("MobilitySimulator object - Vehicle allocation has not been performed.")
        # Check if trip distribution has been performed
        if not 'Origin' in value.flows.columns:
            raise ValueError("MobilitySimulator object - Trip distribution has not been performed.")

        self._mobility_demand = value

    @property
    def scenario(self) -> dict:
        """dict: The charging scenario."""
        return self._scenario

    @scenario.setter
    def scenario(self, value: dict):
        # Define required structure and expected types for each scenario
        required_keys = {
            'home': {
                'share': (float,),  # share should be a float
                'power_options_kW': list,  # power_options should be a list
                'arrival_time_h': list,  # arrival_time should be a list
            },
            'work': {
                'share': (float,),
                'power_options_kW': list,
                'arrival_time_h': list,
            },
            'poi': {
                'share': (float,),
                'power_options_kW': list
            }
        }

        # Validate input value
        if not isinstance(value, dict):
            raise ValueError("scenario must be a dictionary.")

        total_share = 0  # To track the total share for validation

        for key, expected_structure in required_keys.items():
            if key not in value:
                raise ValueError(f"Missing required key: '{key}' in scenario.")

            # Check that each key's value is a dictionary and follows the expected structure
            scenario_part = value[key]
            if not isinstance(scenario_part, dict):
                raise ValueError(f"The value for '{key}' must be a dictionary.")

            # Validate each attribute in the scenario part
            for attr, expected_type in expected_structure.items():
                if attr not in scenario_part:
                    raise ValueError(f"Missing '{attr}' in '{key}' configuration.")
                if not isinstance(scenario_part[attr], expected_type):
                    raise TypeError(f"'{attr}' in '{key}' must be of type {expected_type}.")

            # Sum up the shares and validate they sum to 1
            total_share += scenario_part['share']
            
            # Validate power options format and shares
            power_options = scenario_part['power_options_kW']
            if not all(isinstance(option, list) and len(option) == 2 and 
                       isinstance(option[0], (float, int)) and isinstance(option[1], (float, int)) 
                       for option in power_options):
                raise TypeError(f"Each 'power_options_kW' entry in '{key}' must be a list of [power_level, share].")
            
            # Check that the sum of the shares in power_options is exactly 1
            if sum(option[1] for option in power_options) != 1:
                raise ValueError(f"The sum of shares in 'power_options_kW' for '{key}' must be exactly 1.")
            
        # Final check that total share is equal to 1
        if total_share != 1:
            raise ValueError("The sum of 'share' values across all keys must be exactly 1.")
        
        # If all checks pass, set the value
        self._scenario = value

    @property
    def charging_efficiency(self) -> float:
        """float: The charging efficiency (between 0 and 1)."""
        return self._charging_efficiency

    @charging_efficiency.setter
    def charging_efficiency(self, value: float):
        if value < 0.0 or value > 1.0:
            raise ValueError("The charging efficiency must be between 0 and 1")
        self._charging_efficiency = value

    # Spatial charging demand

    def compute_spatial_demand(self) -> None:
        """Compute the spatial charging demand and store results in the _spatial_demand DataFrame.

        This method calculates the total and per-vehicle daily charging demand at each traffic zone based 
        on the scenario configuration and zone-specific properties. It updates the spatial_demand attribute
        with the computed results.

        Returns:
            None: This method does not return a value; it updates the internal _spatial_demand DataFrame.
        """
        print(f"INFO \t Computing the spatial charging demand...")

        # Step 1: Retrieve the shares for different charging locations (origin, destination, intermediate).
        share_home = self.scenario['home']['share']
        share_work = self.scenario['work']['share']
        share_poi = self.scenario['poi']['share']

        # Step 2: Calculate the fleet-wide average EV consumption 
        average_ev_consumption = self.vehicle_fleet.average_consumption()

        # Step 3: Compute the total and average demand per vehicle at POIs (specific case as we do not know where they charge)
        demand_at_poi = (
            2 * self.mobility_demand.aggregated_zone_metrics['fkt_inflows'].sum() * average_ev_consumption / self.charging_efficiency * share_poi 
        )
        tot_vehicles_at_poi = self.mobility_demand.aggregated_zone_metrics['n_inflows'].sum() * share_poi 
        demand_per_vehicle_at_poi = (demand_at_poi / tot_vehicles_at_poi) if tot_vehicles_at_poi > 0 else 0
        total_pois = self.region.traffic_zones['n_pois'].sum()

        # Step 4: Initialize data storage for zone-specific charging demand.
        data = []

        # Step 5: Loop over each zone to calculate the charging demand at home, work, and pois.
        for index, row in self.mobility_demand.aggregated_zone_metrics.iterrows():
            # Retrieve zone-specific information.
            zone_id = row['id']
            geometry = self.region.traffic_zones.loc[self.region.traffic_zones['id'] == zone_id, 'geometry'].values[0]

            # Step 6: Compute the total energy demand (kWh) for each charging location
            Etot_home = (2 * row['fkt_outflows'] * share_home * average_ev_consumption / self.charging_efficiency)
            Etot_work = (2 * row['fkt_inflows'] * share_work * average_ev_consumption / self.charging_efficiency)

            # For poi, get the number of pois at this taz
            n_pois = self.region.traffic_zones.loc[self.region.traffic_zones['id'] == zone_id, 'n_pois'].values[0]
            Etot_poi = demand_at_poi * (n_pois / total_pois)

            # Step 7: Calculate the number of vehicles charging at home and destination for this TAZ.
            vehicles_home = round(row['n_outflows'] * share_home)
            vehicles_work = round(row['n_inflows'] * share_work)
            vehicles_poi = round(Etot_poi / demand_per_vehicle_at_poi) if Etot_poi > 0 else 0

            # Store calculated vehicles temporarily for adjustment
            data.append({
                'id': zone_id,
                'geometry': geometry,
                'n_vehicles_home': vehicles_home,
                'n_vehicles_work': vehicles_work,
                'n_vehicles_poi': vehicles_poi,
                'Etot_home_kWh': Etot_home,
                'Etot_work_kWh': Etot_work,
                'Etot_poi_kWh': Etot_poi
            })

        # Calculate total vehicles from initial estimates
        total_vehicles = sum(item['n_vehicles_home'] + item['n_vehicles_work'] + item['n_vehicles_poi'] for item in data)

        # Desired total number of vehicles (this value should be set based on your requirements)
        desired_total_vehicles = self.vehicle_fleet.total_vehicles

        difference = desired_total_vehicles - total_vehicles

        if difference != 0:
            print(f"ALERT \t Randomly allocating {difference} vehicles due to rounding errors...")

        # Step 10: Adjust the number of vehicles by adding/removing one vehicle randomly
        while total_vehicles != desired_total_vehicles:
            difference = desired_total_vehicles - total_vehicles  
            
            if difference > 0:  # Adding a vehicle
                # Select a random zone and a random category to add a vehicle
                random_zone_index = np.random.randint(len(data))
                selected_zone = data[random_zone_index]
                
                # Randomly select the category (home, work, poi)
                category = np.random.choice(['n_vehicles_home', 'n_vehicles_work', 'n_vehicles_poi'])
                if not selected_zone[category] == 0:  # Ensure we do not add a vehicle where there is no vehicle
                    selected_zone[category] += 1
                    total_vehicles += 1

            elif difference < 0:  # Removing a vehicle
                # Select a random zone and a random category to remove a vehicle
                random_zone_index = np.random.randint(len(data))
                selected_zone = data[random_zone_index]

                # Randomly select the category (home, work, poi)
                category = np.random.choice(['n_vehicles_home', 'n_vehicles_work', 'n_vehicles_poi'])
                if selected_zone[category] > 0:  # Ensure there's a vehicle to remove
                    selected_zone[category] -= 1
                    total_vehicles -= 1

        # Calculate average charging demand per vehicle for each charging location (kWh per vehicle).
        for item in data:
            item['E_per_vehicle_home_kWh'] = (item['Etot_home_kWh'] / round(item['n_vehicles_home'])) if round(item['n_vehicles_home']) > 0 else 0
            item['E_per_vehicle_work_kWh'] = (item['Etot_work_kWh'] / round(item['n_vehicles_work'])) if round(item['n_vehicles_work']) > 0 else 0
            item['E_per_vehicle_poi_kWh'] = demand_per_vehicle_at_poi if round(item['n_vehicles_poi']) > 0 else 0

        # Step 11: Convert data into a DataFrame and store it in the spatial_demand attribute.
        self._spatial_demand = pd.DataFrame(data)

        # Step 12: Print main aggregated outputs
        print(f" \t > Average consumption: {average_ev_consumption:.3f} kWh/km")
        print(f" \t > Total charging needs: {self._spatial_demand['Etot_home_kWh'].sum() + self._spatial_demand['Etot_work_kWh'].sum() + self._spatial_demand['Etot_poi_kWh'].sum()} kWh")
        print(f" \t > Home: Total: {self._spatial_demand['Etot_home_kWh'].sum():.3f} kWh - "
              f"Per vehicle (weighted avg): {(
                  (self._spatial_demand['E_per_vehicle_home_kWh'] * self._spatial_demand['n_vehicles_home']).sum() /
                  self._spatial_demand['n_vehicles_home'].sum()) if self._spatial_demand['n_vehicles_home'].sum() > 0 else 0:.3f} kWh - "
              f"Vehicles: {self._spatial_demand['n_vehicles_home'].sum()}")

        print(f" \t > Work: Total: {self._spatial_demand['Etot_work_kWh'].sum():.3f} kWh - "
              f"Per vehicle (weighted avg): {(
                  (self._spatial_demand['E_per_vehicle_work_kWh'] * self._spatial_demand['n_vehicles_work']).sum() /
                  self._spatial_demand['n_vehicles_work'].sum()) if self._spatial_demand['n_vehicles_work'].sum() > 0 else 0:.3f} kWh - "
              f"Vehicles: {self._spatial_demand['n_vehicles_work'].sum()}")

        print(f" \t > POIs: Total: {self._spatial_demand['Etot_poi_kWh'].sum():.3f} kWh - "
              f"Per vehicle (weighted avg): {(
                  (self._spatial_demand['E_per_vehicle_poi_kWh'] * self._spatial_demand['n_vehicles_poi']).sum() /
                  self._spatial_demand['n_vehicles_poi'].sum()) if self._spatial_demand['n_vehicles_poi'].sum() > 0 else 0:.3f} kWh - "
              f"Vehicles: {self._spatial_demand['n_vehicles_poi'].sum()}")

    # Temporal charging demand

    def compute_temporal_demand(self, time_step: float, travel_time_home_work: float = 0.5, soc_threshold_mean: float = 0.6, soc_threshold_std_dev: float = 0.2):
        """        
        Computes the temporal charging demand for vehicles, assigning their properties 
        and calculating the charging profile based on specified parameters.

        Parameters:
            time_step (float): The time interval (in hours) for the charging profile computation.
            travel_time_home_work (float): Average travel time from home to work (in hours).
            soc_threshold_mean (float): Mean state of charge (SoC) threshold for vehicle charging.
            soc_threshold_std_dev (float): Standard deviation for the SoC threshold.

        Returns:
            None
        """
        print(f"INFO \t Computing the temporal charging demand...")

        if self._spatial_demand is None: 
            raise RuntimeError(f"ERROR \t Please compute the spatial charging demand before the temporal one.")

        self._assign_vehicle_properties(travel_time_home_work, soc_threshold_mean, soc_threshold_std_dev)
        self._compute_charging_profile(time_step)
        self._compute_aggregated_charging_profile()

    def _assign_vehicle_properties(self, travel_time_home_work: float = 0.5, soc_threshold_mean: float = 0.6, soc_threshold_std_dev: float = 0.2):
        """
        Assigns energy demand, arrival times, and other properties to each vehicle 
        based on the charging location, including calculations for daily charging demand 
        and vehicle characteristics.

        Parameters:
            travel_time_home_work (float): Average travel time from home to work (in hours).
            soc_threshold_mean (float): Mean state of charge (SoC) threshold for vehicle charging.
            soc_threshold_std_dev (float): Standard deviation for the SoC threshold.

        Returns:
            None
        """

        # Step 1: Assign vehicle properties
        ########################################################
        print(f"INFO \t Assigning properties to each vehicle...")

        all_vehicle_properties = []  # To store DataFrames for each location
        vehicle_id_counter = 0  # Initialize a counter for unique vehicle IDs

        # Loop through all possible charging locations
        for charging_location in self.scenario.keys():

            # Get the number of vehicles and scenario settings
            vehicle_counts = self.spatial_demand['n_vehicles_' + charging_location].sum()
            # Skip if there are no vehicles for this charging location
            if vehicle_counts == 0:
                continue

            scenario = self.scenario[charging_location]
            work_arrival_mean, work_arrival_std = self.scenario['work']['arrival_time_h']
            home_arrival_mean, home_arrival_std = self.scenario['home']['arrival_time_h']

            # Assign origin zone to vehicles 
            origin_flows = self.mobility_demand.flows.groupby("Origin")["Flow"].sum().reset_index()
            zone_probabilities = origin_flows["Flow"] / origin_flows["Flow"].sum()
            assigned_zones = np.random.choice(origin_flows["Origin"], size=vehicle_counts, p=zone_probabilities)
            
            # Initialize a DataFrame for vehicle properties
            vehicle_properties = pd.DataFrame({
                "vehicle_id": np.arange(vehicle_id_counter, vehicle_id_counter + vehicle_counts),  # Unique vehicle IDs
                "name": np.empty(vehicle_counts, dtype=object),
                "location": charging_location,
                "origin_zone": assigned_zones,
                "days_between_charges": np.zeros(vehicle_counts),
                "charging_demand": np.zeros(vehicle_counts),
                "arrival_time": np.zeros(vehicle_counts),
                "departure_time": np.zeros(vehicle_counts),
                "idling_duration": np.zeros(vehicle_counts),
                "charging_power": np.zeros(vehicle_counts),
                "strategy": np.zeros(vehicle_counts),                                
            })

            # Update the vehicle counter to ensure unique IDs
            vehicle_id_counter += vehicle_counts

            # Loop over each zone to calculate zone-specific travel distance distributions
            selected_distances = np.zeros(vehicle_counts)  # Placeholder for distances
            for zone in np.unique(assigned_zones):
                zone_indices = np.where(assigned_zones == zone)[0]  # Vehicles in this zone
                zone_demand = self.mobility_demand.flows[self.mobility_demand.flows['Origin'] == zone]
                
                # Aggregate flows and calculate distances and probabilities for this zone
                grouped_zone_demand = zone_demand.groupby('Travel Distance (km)')['Flow'].sum().reset_index()
                grouped_zone_demand['Probability'] = grouped_zone_demand['Flow'] / grouped_zone_demand['Flow'].sum()
                distances = grouped_zone_demand['Travel Distance (km)'].values
                probabilities = grouped_zone_demand['Probability'].values
                
                # Assign travel distances for vehicles in this zone
                selected_distances[zone_indices] = np.random.choice(distances, size=len(zone_indices), p=probabilities)

            # Randomly select vehicle types for all vehicles at once
            vehicle_types, vehicle_shares = zip(*self.vehicle_fleet.vehicle_types)
            selected_vehicles = np.random.choice(vehicle_types, size=vehicle_counts, p=vehicle_shares)

            # Populate the vehicle names directly
            vehicle_properties['name'] = [vehicle.name for vehicle in selected_vehicles]
            vehicle_properties['strategy'] = ["dumb" for vehicle in selected_vehicles]        
            
            # Calculate daily charging demand in a vectorized way
            daily_charging_demand = 2 * selected_distances * np.array([vehicle.consumption_kWh_per_km for vehicle in selected_vehicles]) / self.charging_efficiency

            vehicle_properties['days_between_charges'] = np.vectorize(hlp.calculate_days_between_charges_single_vehicle)(
                daily_charging_demand, 
                np.array([vehicle.battery_capacity_kWh for vehicle in selected_vehicles]),
                soc_threshold_mean * np.ones(vehicle_counts),
                soc_threshold_std_dev * np.ones(vehicle_counts)
            )

            # Calculate charging demand in a vectorized way
            vehicle_properties['charging_demand'] = daily_charging_demand * vehicle_properties['days_between_charges']
            
            # Prepare power options
            power_options = np.array(scenario['power_options_kW'])
            max_charging_powers = np.array([vehicle.max_charging_power_kW for vehicle in selected_vehicles])
            
            # Select valid charging powers below the max charging power for each vehicle
            valid_power_choices = [
                [power for power, _ in power_options if power <= max_power]
                for max_power in max_charging_powers
            ]
            
            # Randomly select charging power for each vehicle
            vehicle_properties['charging_power'] = [
                random.choice(valid_powers) if valid_powers else None for valid_powers in valid_power_choices
            ]
            
            # Handle arrival and departure times based on charging location
            if charging_location != 'poi':
                arrival_times = np.random.normal(loc=scenario['arrival_time_h'][0], scale=scenario['arrival_time_h'][1], size=vehicle_counts)
                vehicle_properties['arrival_time'] = arrival_times % 24  # Use modulo for wrapping
            else:
                random_arrival = np.random.normal(work_arrival_mean, work_arrival_std / 2, size=vehicle_counts)
                random_departure = np.random.normal(home_arrival_mean, home_arrival_std / 2, size=vehicle_counts)
                vehicle_properties['arrival_time'] = (np.random.uniform(random_arrival, random_departure) % 24)

            if charging_location == 'home':
                work_arrival_times = np.random.normal(work_arrival_mean, work_arrival_std, size=vehicle_counts)
                vehicle_properties['departure_time'] = (work_arrival_times - travel_time_home_work) % 24  # Wrap with modulo
            elif charging_location == 'work':
                home_arrival_times = np.random.normal(home_arrival_mean, home_arrival_std, size=vehicle_counts)
                vehicle_properties['departure_time'] = (home_arrival_times - travel_time_home_work) % 24
            elif charging_location == 'poi':
                charging_durations = vehicle_properties['charging_demand'] / vehicle_properties['charging_power']
                vehicle_properties['departure_time'] = (vehicle_properties['arrival_time'] + charging_durations) % 24  # Wrap with modulo

            # Vectorized calculation of idling time
            arrival_times = vehicle_properties['arrival_time'].values
            departure_times = vehicle_properties['departure_time'].values

            # Calculate idling time based on conditions
            vehicle_properties['idling_duration'] = np.where(
                departure_times >= arrival_times,
                departure_times - arrival_times,
                (24 - arrival_times) + departure_times
            )

            # Append the DataFrame to the list
            all_vehicle_properties.append(vehicle_properties)

            # Compute statistics for this scenario
            num_vehicles = len(vehicle_properties)
            avg_demand= vehicle_properties['charging_demand'].mean()
            days= vehicle_properties['days_between_charges'].mean()

            vehicle_counts = vehicle_properties['name'].value_counts().to_dict()

        # Step 2: Select and filter only vehicles charging today
        ########################################################
        print(f"INFO \t Selecting and filtering only vehicles charging today...")

        # Concatenate all location-specific DataFrames into a single DataFrame
        vehicle_properties = pd.concat(all_vehicle_properties, ignore_index=True)

        charging_probability = 1 / vehicle_properties['days_between_charges']
        
        # Determine if each vehicle will charge today using vectorized random sampling
        charging_today = np.random.rand(len(charging_probability)) <= charging_probability
        
        # Modify the original DataFrame to keep only vehicles charging today
        vehicle_properties = vehicle_properties[charging_today]
        
        # Reset the index for the modified DataFrame
        vehicle_properties.reset_index(drop=True, inplace=True)

        print(f"\t > Number of vehicles charging: {len(vehicle_properties)}")

        self._temporal_demand_vehicle_properties = vehicle_properties

    def _compute_charging_profile(self, time_step: float):
        """
        Creates a 24-hour charging profile for each vehicle, detailing the charging power 
        at each specified time step.

        Parameters:
            time_step (float): The time step interval (in hours) for which the charging profile is computed.

        Returns:
            None
        """
        print(f"INFO \t Computing the charging profile for each vehicle...")
        vehicle_properties = self.temporal_demand_vehicle_properties

        # Define the time intervals
        num_time_steps = int(24 / time_step)
        time_steps = [(i * time_step) for i in range(num_time_steps)]
        
        # Create a list to hold individual vehicle charging profiles
        profiles = []

        # Precompute relevant values
        charging_powers = vehicle_properties["charging_power"].values
        arrival_times = vehicle_properties["arrival_time"].values
        idling_durations = vehicle_properties["idling_duration"].values
        charging_demands = vehicle_properties["charging_demand"].values

        # Calculate charging durations and indices
        charging_durations = charging_demands / charging_powers
        start_indices = np.round(arrival_times / time_step).astype(int)
        end_indices = start_indices + np.round(charging_durations / time_step).astype(int)

        # Preliminary checks
        if np.any(charging_durations <= time_step):
            num_alerts = np.sum(charging_durations <= time_step)
            print(f"ALERT \t {num_alerts}  charging durations are smaller than the timestep. Charging demand may not be met.")
        
        # Check how many vehicles have a charging duration greater than their idling time
        vehicles_with_long_charging = (charging_durations - 0.01) > idling_durations # Adding a small correction to account for rounding issues
        count_long_charging = vehicles_with_long_charging.sum()
        if count_long_charging > 0:
            print(f"ALERT \t {count_long_charging} vehicle(s) require charging durations longer than their idling periods. Charging continues beyond the expected idling time.")

        # Process each vehicle
        for idx, vehicle_id in enumerate(vehicle_properties["vehicle_id"]):
            # Initialize profile with zeros
            profile = np.zeros(num_time_steps)
            
            if end_indices[idx] >= num_time_steps:  # Wrap-around case
                # Charging from start index to the end of the day
                profile[start_indices[idx]:] += charging_powers[idx]
                # Charging from the start of the day to the wrap end
                wrap_end = end_indices[idx] % num_time_steps
                profile[:wrap_end] += charging_powers[idx]
            else:  # No-wrap case
                profile[start_indices[idx]:end_indices[idx]] += charging_powers[idx]

            # Append the vehicle profile to the profiles list
            profiles.append([vehicle_id] + profile.tolist())

        # Create DataFrame from profiles
        profile_df = pd.DataFrame(profiles, columns=["vehicle_id"] + time_steps)

        # Transpose the DataFrame and reset the index
        profile_df = profile_df.set_index('vehicle_id').transpose().reset_index()
        profile_df.columns = ['time'] + profile_df.columns[1:].tolist()  # Rename first column to 'time'

        self._temporal_demand_profile = profile_df

    def _compute_aggregated_charging_profile(self):
        """
        Creates an aggregated charging profile, summing the charging power of all vehicles 
        across different locations (home, work, poi) for each time step. It also includes 
        statistics on the number of vehicles plugged in and charging at each location.

        Parameters:
            time_step (float): The time step interval (in hours) for which the aggregated charging profile is computed.

        Returns:
            None
        """
        print("INFO \t Computing the aggregated charging profile by location...")

        # Get the individual charging profile and vehicle properties DataFrames
        vehicle_properties = self.temporal_demand_vehicle_properties  # contains 'vehicle_id', 'arrival_time', 'departure_time', 'location'
        charging_profile = self.temporal_demand_profile               # contains 'vehicle_id', 'time', and power data

        # Convert time columns to float (if not already)
        time_step = charging_profile['time'][1] - charging_profile['time'][0]
        vehicle_properties['arrival_time'] = vehicle_properties['arrival_time'].astype(float)
        vehicle_properties['departure_time'] = vehicle_properties['departure_time'].astype(float)
        
        # Reshape the charging profile to long format with each vehicle's charging power at each time interval
        melted_profile = charging_profile.melt(id_vars='time', var_name='vehicle_id', value_name='charging_power')
        melted_profile['vehicle_id'] = melted_profile['vehicle_id'].astype(int)

        # Merge with vehicle properties to associate each vehicle's charging with its location
        merged_df = melted_profile.merge(vehicle_properties[['vehicle_id', 'location', 'arrival_time', 'departure_time']], on='vehicle_id')

        # Vectorized determination of whether each vehicle is plugged in based on arrival and departure times
        arrival_time = merged_df['arrival_time']
        departure_time = merged_df['departure_time']
        time = merged_df['time']
        
        merged_df['is_plugged'] = ((departure_time > arrival_time) & (arrival_time <= time) & (time < departure_time)) | \
                                  ((departure_time <= arrival_time) & ((time < departure_time) | (time >= arrival_time)))

        # Aggregate charging power by time and location
        aggregated_profile = (merged_df
                              .groupby(['time', 'location'])['charging_power']
                              .sum()
                              .unstack(fill_value=0)
                              .reset_index()
                              .rename_axis(None, axis=1))

        # Ensure all required columns are present
        for col in ['home', 'work', 'poi']:
            if col not in aggregated_profile.columns:
                aggregated_profile[col] = 0  # Add missing location columns with zeroes if no data for location

        # Add the 'total' column summing across 'home', 'work', and 'poi'
        aggregated_profile['total'] = aggregated_profile[['home', 'work', 'poi']].sum(axis=1)

        # Count the number of vehicles charging (non-zero power) by location and time
        vehicle_charging_counts = (merged_df[merged_df['charging_power'] > 0]
                                   .groupby(['time', 'location'])['vehicle_id']
                                   .nunique()
                                   .unstack(fill_value=0)
                                   .reset_index())
        
        # Ensure all required columns are present in vehicle_charging_counts
        for col in ['home', 'work', 'poi']:
            if col not in vehicle_charging_counts.columns:
                vehicle_charging_counts[col] = 0

        # Rename columns to indicate they are counts of charging vehicles
        vehicle_charging_counts = vehicle_charging_counts.rename(columns={'home': 'home_vehicle_charging',
                                                                          'work': 'work_vehicle_charging',
                                                                          'poi': 'poi_vehicle_charging'})
        
        # Add a total vehicle charging count column
        vehicle_charging_counts['total_vehicle_charging'] = vehicle_charging_counts[['home_vehicle_charging', 
                                                                                    'work_vehicle_charging', 
                                                                                    'poi_vehicle_charging']].sum(axis=1)

        # Count the number of vehicles plugged in (regardless of charging status) by location and time
        vehicle_plugged_counts = (merged_df[merged_df['is_plugged']]
                                  .groupby(['time', 'location'])['vehicle_id']
                                  .nunique()
                                  .unstack(fill_value=0)
                                  .reset_index())
        
        # Ensure all required columns are present in vehicle_plugged_counts
        for col in ['home', 'work', 'poi']:
            if col not in vehicle_plugged_counts.columns:
                vehicle_plugged_counts[col] = 0

        # Rename columns to indicate they are counts of plugged-in vehicles
        vehicle_plugged_counts = vehicle_plugged_counts.rename(columns={'home': 'home_vehicle_plugged',
                                                                        'work': 'work_vehicle_plugged',
                                                                        'poi': 'poi_vehicle_plugged'})
        
        # Add a total plugged-in vehicle count column
        vehicle_plugged_counts['total_vehicle_plugged'] = vehicle_plugged_counts[['home_vehicle_plugged', 
                                                                                  'work_vehicle_plugged', 
                                                                                  'poi_vehicle_plugged']].sum(axis=1)

        # Merge the counts back into the aggregated profile
        aggregated_profile = aggregated_profile.merge(vehicle_charging_counts, on='time', how='left').fillna(0)
        aggregated_profile = aggregated_profile.merge(vehicle_plugged_counts, on='time', how='left').fillna(0)

        self._temporal_demand_profile_aggregated = aggregated_profile[['time', 'home', 'work', 'poi', 'total',
                                                                       'home_vehicle_plugged', 'work_vehicle_plugged', 
                                                                       'poi_vehicle_plugged', 'total_vehicle_plugged',
                                                                       'home_vehicle_charging', 'work_vehicle_charging', 
                                                                       'poi_vehicle_charging', 'total_vehicle_charging']]

        # Convert power (kW) to energy (kWh) by multiplying each time step by time_step (in hours)
        energy_needs = aggregated_profile[['home', 'work', 'poi', 'total']] * time_step

        # Calculate total and per-location charging needs (kWh)
        total_charging_needs = energy_needs['total'].sum()
        home_charging_needs = energy_needs['home'].sum()
        work_charging_needs = energy_needs['work'].sum()
        poi_charging_needs = energy_needs['poi'].sum()

        print(f"\t > Total Charging Needs: {total_charging_needs:.2f} kWh")
        print(f"\t > Home Charging Needs: {home_charging_needs:.2f} kWh")
        print(f"\t > Work Charging Needs: {work_charging_needs:.2f} kWh")
        print(f"\t > POI Charging Needs: {poi_charging_needs:.2f} kWh")

    # Smart charging

    def apply_smart_charging(self, location: list, charging_strategy: str, share: float, **kwargs):
        """
        Apply a smart charging strategy to a subset of vehicles at specific locations.
        
        Args:
            location (list): List of location types ('home', 'work', 'poi').
            charging_strategy (str): The name of the charging strategy to apply.
            share (float): Proportion of vehicles participating (between 0 and 1).
            kwargs: Additional parameters for specific charging strategies.
        """
        print(f"INFO \t Applying '{charging_strategy}' charging strategy...")

        # Check that share is within the valid range
        if not (0 <= share <= 1):
            raise ValueError("Share must be between 0 and 1.")
        
        selected_vehicle_ids = []  # Initialize an empty list for selected vehicle IDs

        # Iterate through each specified location
        for l in location:
            # Filter vehicles by the current location
            location_vehicles = self._temporal_demand_vehicle_properties[self._temporal_demand_vehicle_properties['location'] == l]
            
            # Determine the number of vehicles to modify for the current location
            num_smart_vehicles = int(share * len(location_vehicles))
            
            # If there are vehicles at this location, select some
            if num_smart_vehicles > 0:
                # Ensure that we do not attempt to select more vehicles than available
                selected_vehicles = random.sample(list(location_vehicles['vehicle_id']), min(num_smart_vehicles, len(location_vehicles)))
                selected_vehicle_ids.extend(selected_vehicles)  # Add to the total list of selected vehicle IDs
                
                print(f"\t > Selected {len(selected_vehicles)} vehicles from '{l}'")

        # Check if there are any vehicles to process
        if not selected_vehicle_ids:
            print(">\t No vehicles found for the given locations.")
            return
        
        # Update the strategy column for selected and non-selected vehicles
        self._temporal_demand_vehicle_properties['strategy'] = self._temporal_demand_vehicle_properties[
            'vehicle_id'].apply(lambda vid: charging_strategy if vid in selected_vehicle_ids else "dumb")

        # Get relevant columns in _temporal_demand_profile
        smart_vehicle_columns = [vid for vid in selected_vehicle_ids if vid in self._temporal_demand_profile.columns]
        smart_vehicles_df = self._temporal_demand_profile[['time'] + smart_vehicle_columns].copy()
        
        # Apply the selected charging strategy
        modified_smart_vehicles_df = self._apply_charging_strategy(smart_vehicles_df, charging_strategy, **kwargs)
        
        # Update _temporal_demand_profile with modified charging profiles
        self._temporal_demand_profile.update(modified_smart_vehicles_df)
        
        # Recompute aggregated profile
        self._compute_aggregated_charging_profile()

    def _apply_charging_strategy(self, smart_vehicles_df: pd.DataFrame, strategy: str, **kwargs) -> pd.DataFrame:
        """
        Applies a specific charging strategy to the smart vehicles.

        Args:
            smart_vehicles_df (pd.DataFrame): DataFrame of vehicles to apply the strategy on.
            strategy (str): The name of the charging strategy.
            **kwargs: Additional parameters specific to certain strategies.

        Returns:
            pd.DataFrame: Modified DataFrame with updated charging profiles.
        """

        # Multiply by (dummy strategy) 
        ##############################
        if strategy == "multiply_by":
            factor = kwargs.get("factor")
            # Multiply each vehicle's charging profile by factor (skip the 'time' column)
            for column in smart_vehicles_df.columns[1:]:  # Skip the 'time' column
                smart_vehicles_df[column] *= factor

        # Peak shaving through ideal coordination 
        #########################################
        elif strategy == "peak_shaving":

            # Initialize peak power and power demand array
            peak_power = 0  # Total power peak
            time = smart_vehicles_df.iloc[:, 0].values
            time_step = time[1] - time[0]
            power_demand = np.zeros(len(time))  # Tracks power demand over time intervals

            # Initialize charging profile for each vehicle
            smart_charging_profile = pd.DataFrame(0, index=time, columns=smart_vehicles_df.columns[1:], dtype=float)

            # Filter for vehicles by vehicle_id in the smart_vehicle_ids list
            smart_vehicle_ids = smart_vehicles_df.columns[1:]
            smart_vehicles_props = self.temporal_demand_vehicle_properties[self.temporal_demand_vehicle_properties['vehicle_id'].isin(smart_vehicle_ids)]

            # Process each vehicle
            for i, row in smart_vehicles_props.iterrows():
                # print(i)
                vehicle_id = row['vehicle_id']
                arrival_time = row['arrival_time']
                departure_time = row['departure_time']
                charging_demand = row['charging_demand']
                max_power = row['charging_power']

                # Calculate start and end indices in the time array for charging window
                start_idx = np.searchsorted(time, arrival_time)
                end_idx = np.searchsorted(time, departure_time)
                if end_idx < start_idx:                    
                    end_idx += len(time)  # Handle wrap-around for overnight charging

                # Remaining demand to be met
                remaining_demand = charging_demand

                # Minimize the total peak load
                for t in range(start_idx, end_idx):
                    current_time_idx = t % len(time)  # Wrap around 24-hour period
                    current_total_power = power_demand[current_time_idx]

                    # If the current total power is below the peak power, charge with maximum power to stay below the limit
                    if current_total_power < peak_power:
                        charge_power = min(peak_power - current_total_power, max_power)
                        charge_energy = charge_power * time_step

                        # Ensure not to exceed remaining charging demand
                        if charge_energy > remaining_demand:
                            charge_energy = remaining_demand
                            charge_power = charge_energy / time_step

                        power_demand[current_time_idx] += charge_power
                        remaining_demand -= charge_energy
                        smart_charging_profile.at[time[current_time_idx], vehicle_id] = charge_power

                        # Stop charging if demand is fully met
                        if remaining_demand <= 0:
                            break

                # If there’s still demand, distribute uniformly across the charging window
                if remaining_demand > 0:
                    charging_duration = (24 - arrival_time + departure_time) if departure_time < arrival_time else (departure_time - arrival_time)
                    charge_power = remaining_demand / charging_duration

                    for t in range(start_idx, end_idx):
                        current_time_idx = t % len(time)
                        power_demand[current_time_idx] += charge_power
                        smart_charging_profile.at[time[current_time_idx], vehicle_id] += charge_power

                # Update peak power if needed
                peak_power = max(power_demand)

            # Reset the index to convert the time index into a column
            smart_charging_profile.reset_index(drop=False, inplace=True)

            # Rename the 'index' column to 'time' to reflect its content
            smart_charging_profile.rename(columns={'index': 'time'}, inplace=True)

            smart_vehicles_df = smart_charging_profile
        else:
            raise ValueError(f"Charging strategy is unknown.")

        # Additional strategies can be implemented here

        return smart_vehicles_df

    # Export and visualization

    def to_csv(self, filepath: str):
        """
        Saves two CSV files with the main output data for flows and aggregated zone metrics.

        Args:
            filepath (str): Base path for the output files. Automatically appends suffixes
                            "_flows" and "_aggregated_zone_metrics" before ".csv".
        """
        # Remove any existing file extension
        filepath_without_ext, _ = os.path.splitext(filepath)

        # Save each dataframe with the respective suffix
        self._spatial_demand.to_csv(f"{filepath_without_ext}_spatial_demand.csv")
        self._temporal_demand_vehicle_properties.to_csv(f"{filepath_without_ext}_temporal_demand_vehicle_properties.csv")
        self._temporal_demand_profile.to_csv(f"{filepath_without_ext}_temporal_demand_profile.csv")
        self._temporal_demand_profile_aggregated.to_csv(f"{filepath_without_ext}_temporal_demand_profile_aggregated.csv")

    def chargingdemand_total_to_map(self, filepath: str):
        """
        Creates a Folium map visualizing the total charging demand at origin and destination points.S
        """
        df = self.spatial_demand

        # 1. Create base map with administrative boundaries
        m = hlp.create_base_map(self.region)

        # 2. Add TAZ boundaries
        hlp.add_taz_boundaries(m, self.region.traffic_zones)

        # 3. Add color-mapped feature groups for charging demand
        hlp.add_colormapped_feature_group(m, df, self.region, 'Etot_home_kWh', 'Charging demand at Home', 'Charging demand (kWh)')
        hlp.add_colormapped_feature_group(m, df, self.region, 'Etot_work_kWh', 'Charging demand at Work', 'Charging demand (kWh)')
        hlp.add_colormapped_feature_group(m, df, self.region, 'Etot_poi_kWh', 'Charging demand at POIs', 'Charging demand (kWh)')

        # 4. Add Layer Control and Save the map
        folium.LayerControl().add_to(m)
        m.save(filepath)

    def chargingdemand_pervehicle_to_map(self, filepath: str):
        """
        Creates a Folium map visualizing the charging demand per vehicle at home, work, and POI.
        """
        df = self.spatial_demand

        # 1. Create base map with administrative boundaries
        m = hlp.create_base_map(self.region)

        # 2. Add TAZ boundaries
        hlp.add_taz_boundaries(m, self.region.traffic_zones)

        # 3. Add color-mapped feature groups for charging demand per vehicle
        hlp.add_colormapped_feature_group(m, df, self.region, 'E_per_vehicle_home_kWh', 'Charging need per vehicle at Home', 'Charging demand (kWh/vehicle)')
        hlp.add_colormapped_feature_group(m, df, self.region, 'E_per_vehicle_work_kWh', 'Charging need per vehicle at Work', 'Charging demand (kWh/vehicle)')
        hlp.add_colormapped_feature_group(m, df, self.region, 'E_per_vehicle_poi_kWh', 'Charging need per vehicle at POIs', 'Charging demand (kWh/vehicle)')

        # 4. Add Layer Control and Save the map
        folium.LayerControl().add_to(m)
        m.save(filepath)

    def chargingdemand_nvehicles_to_map(self, filepath: str):
        """
        Creates a Folium map visualizing the number of vehicles charging at home, work, and POI locations.
        """
        df = self.spatial_demand

        # 1. Create base map with administrative boundaries
        m = hlp.create_base_map(self.region)

        # 2. Add TAZ boundaries
        hlp.add_taz_boundaries(m, self.region.traffic_zones)

        # 3. Add color-mapped feature groups for number of vehicles charging
        hlp.add_colormapped_feature_group(m, df, self.region, 'n_vehicles_home', 'Number of vehicles charging at Home', 'Number of vehicles')
        hlp.add_colormapped_feature_group(m, df, self.region, 'n_vehicles_work', 'Number of vehicles charging at Work', 'Number of vehicles')
        hlp.add_colormapped_feature_group(m, df, self.region, 'n_vehicles_poi', 'Number of vehicles charging at POIs', 'Number of vehicles')

        # 4. Add Layer Control and Save the map
        folium.LayerControl().add_to(m)
        m.save(filepath)