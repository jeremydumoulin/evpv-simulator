# coding: utf-8

import json
import os
import rasterio
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import shape, LineString, Point, Polygon, box, MultiPoint
from shapely.ops import transform, nearest_points, snap
from geopy.distance import geodesic, distance
import openrouteservice
import time
import math
import csv
import pickle
import folium
import branca.colormap as cm

from evpv.vehicle import Vehicle
from evpv.vehiclefleet import VehicleFleet
from evpv.region import Region
from evpv.mobilitysimulator import MobilitySimulator
from evpv.pvsimulator import PVSimulator

class ChargingSimulator:
    """  
    A class to simulate the daily charging demand.
    """
    def __init__(self, region: Region, vehicle_fleet: VehicleFleet, mobility_demand: MobilitySimulator, scenario: dict, charging_efficiency: float):
        """
        Initializes the ChargingDemandSimulator class.
        """
        print("=========================================")
        print(f"INFO \t Creation of a MobilitySimulator object.")
        print("=========================================")

        self.vehicle_fleet = vehicle_fleet
        self.region = region
        self.mobility_demand = mobility_demand   
        self.scenario = scenario
        self.charging_efficiency = charging_efficiency       

        print(f"INFO \t Successful initialization of input parameters.")

        # Modeling results
        self._spatial_demand = pd.DataFrame()
        self._temporal_demand = pd.DataFrame() 

    # Results properties (read-only)

    @property
    def spatial_demand(self) -> pd.DataFrame:
        """pd.DataFrame: The spatial charging demand."""
        return self._spatial_demand

    @property
    def temporal_demand(self) -> pd.DataFrame:
        """pd.DataFrame: The temporal charging demand."""
        return self._temporal_demand

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
                'power_options_kW': list,
                'arrival_time_h': list,
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
            
            # Validate arrival times are between 0 and 24
            arrival_time = scenario_part['arrival_time_h']
            if not all(isinstance(time, (float, int)) for time in arrival_time):
                raise TypeError(f"'arrival_time_h' in '{key}' must be a list of floats or integers.")
            if not all(0 <= time <= 24 for time in arrival_time):
                raise ValueError(f"All 'arrival_time_h' values in '{key}' must be between 0 and 24.")

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
            print(f"ALERT \t Randomly allocating {difference} vehicles due to rounding erros...")

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

    def eval_charging_profile(self, charging_location: str, time_step: float) -> np.ndarray:
        """
        Evaluates the individual charging profiles at a given location for each vehicle
        that is charging, using filtered vehicle properties for simplicity.

        Args:
            charging_location (str): Location where vehicles will be charging ('home', 'work', 'poi').
            time_step (float): Time step for charging profile in hours.

        Returns:
            np.ndarray: 2D array where each row represents the charging profile of a single charging vehicle.
        """
        # Step 1: Assign vehicle properties
        vehicle_properties = self.assign_vehicle_properties(charging_location)

        # Step 2: Select and filter only vehicles charging today
        vehicle_properties = self.select_charging_vehicles(vehicle_properties, soc_threshold)

        # Step 3: Assign charging powers based on the available power options the maximum charging power of each vehicle (drop vehicle if charging time is not enough)
        vehicle_properties = self.assign_charging_power(vehicle_properties, charging_location)

        # Step 4: Compute charging profile for each charging vehicle
        num_charging_vehicles = len(vehicle_properties['energy_demand'])
        all_profiles = []

        for i in range(num_charging_vehicles):
            profile = self.get_single_vehicle_charging_profile(
                vehicle_properties['arrival_times'][i],
                vehicle_properties['departure_times'][i],
                vehicle_properties['energy_demand'][i],
                vehicle_properties['charging_powers'][i],
                time_step
            )
            all_profiles.append(profile)

        return np.array(all_profiles)

    def assign_vehicle_properties(self, charging_location: str):
        """
        Assigns energy demand, arrival times, and other properties to each vehicle
        based on the charging location.

        Args:
            charging_location (str): Location where vehicles will be charging ('home', 'work', 'poi').

        Returns:
            dict: Contains arrays of vehicle properties like energy_demand, arrival_times, etc.
        """
        vehicle_counts = self.charging_demand['n_vehicles_' + charging_location].sum()

        # Initialize vehicle property arrays
        vehicle_properties = {
            "energy_demand": np.zeros(vehicle_counts),
            "arrival_times": np.zeros(vehicle_counts),
            "departure_times": np.zeros(vehicle_counts),
            "days_between_charges": np.zeros(vehicle_counts),
            # Additional properties can be added as needed
        }

        # Populate properties based on the fleet composition
        # Example: vehicle_properties['energy_demand'] = <your logic here>

        return vehicle_properties

    def select_charging_vehicles(self, vehicle_properties: dict):
        """
        Randomly selects vehicles that will charge today based on charging probability.

        Args:
            vehicle_properties (dict): Dictionary of assigned vehicle properties.

        Returns:
            np.ndarray: Boolean array indicating which vehicles are charging today.
        """
        charging_probability = 1 / vehicle_properties['days_between_charges']
        charging_today = np.random.rand(len(charging_probability)) <= charging_probability
        return charging_today

    def assign_charging_power(self, vehicle_properties: dict, charging_today: np.ndarray, charging_location: str):
        """
        Assigns a random charging power to vehicles based on location and ensures feasible charging durations.

        Args:
            vehicle_properties (dict): Dictionary of assigned vehicle properties.
            charging_today (np.ndarray): Boolean array of vehicles charging today.
            charging_location (str): Location where vehicles will be charging ('home', 'work', 'poi').

        Returns:
            dict: Updated vehicle properties with charging powers and feasible charging durations.
        """
        # Assign charging power and validate durations
        for i, charging in enumerate(charging_today):
            if not charging:
                continue
            # Example: vehicle_properties['charging_powers'][i] = <your logic here>
            # Validate the charging time and adjust if needed

        return vehicle_properties

    def get_single_vehicle_charging_profile(self, arrival_time, departure_time, charging_demand, charging_power):
        """
        Computes the charging profile for a single vehicle.

        Args:
            arrival_time (float): Arrival time of the vehicle in hours.
            departure_time (float): Departure time of the vehicle in hours.
            charging_demand (float): Energy demand in kWh.
            charging_power (float): Charging power in kW.

        Returns:
            np.ndarray: Charging profile array over the time period.
        """
        time_points = int(24 / self.time_step)
        charging_profile = np.zeros(time_points)

        # Calculate start and end indices for charging
        start_idx = int(arrival_time / self.time_step)
        duration_hours = charging_demand / charging_power
        end_idx = int((arrival_time + duration_hours) / self.time_step) % time_points

        # Apply charging power within the duration, considering wrap-around if necessary
        if end_idx > start_idx:
            charging_profile[start_idx:end_idx] = charging_power
        else:
            charging_profile[start_idx:] = charging_power
            charging_profile[:end_idx] = charging_power

        return charging_profile

