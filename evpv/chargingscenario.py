# coding: utf-8

import numpy as np
import pandas as pd
import random
import geopandas as gpd
import json
import warnings
import os
import rasterio
from rasterio.mask import mask
from pathlib import Path
from shapely.geometry import shape, LineString, Point, Polygon, box, MultiPoint
from shapely.ops import transform, nearest_points, snap
import pyproj
from pyproj import Geod
import math
import matplotlib.pyplot as plt
import folium
import branca.colormap as cm
from scipy.stats import beta

from evpv import helpers as hlp
from evpv.mobilitysim import MobilitySim

class ChargingScenario:
    """  
    A class to simulate the daily charging demand using mobility demand results and assumptions 
    regarding the charging scenario, the EV fleet and charging powers.
    """

    #######################################
    ############# Constructor #############
    #######################################

    def __init__(self, mobsim: list[MobilitySim], ev_fleet: list[dict], charging_efficiency: float, time_step: float, scenario_definition: dict):
        """
        Initializes the ChargingScenario class.

        Args:
            mobsim (list[MobilitySim]): A list of MobilitySim objects.
            ev_fleet (list[dict]): A list of electric vehicle fleet details.
            charging_efficiency (float): The efficiency of EV charging (between 0 and 1).
            time_step (float): The time step for charging simulations.
            scenario_definition (dict): A dictionary defining the charging scenario.

        Raises:
            ValueError: If any input parameters are out of expected ranges.
        """
        self.mobsim = mobsim        
        self.ev_fleet = ev_fleet
        self.charging_efficiency = charging_efficiency 
        self.taz_properties = mobsim  # Combine TAZ properties of each mobsim into a single effective TAZ properties df       
        self.time_step = time_step
        self.scenario_definition = scenario_definition

        # Modeling results
        self._charging_demand = pd.DataFrame()
        self._charging_profile = pd.DataFrame()    

        # Printing results
        print(f"INFO \t New ChargingDemand object created")
        print(f" \t Number of vehicles: {self.taz_properties['n_inflows'].sum()} (n_in) // {self.taz_properties['n_outflows'].sum()} (n_out)")
        print(f" \t FKT (origin to destination): {self.taz_properties['fkt_inflows'].sum()} km")
        print(f" \t Average VKT (origin to destination): {self.taz_properties['fkt_inflows'].sum() / self.taz_properties['n_inflows'].sum()} km")  

    #######################################
    ###### Main Setters and Getters #######
    #######################################

    # Mobism
    @property
    def mobsim(self) -> list[MobilitySim]:
        """Get the mobility simulation object.

        Returns:
            list[MobilitySim]: The mobility simulation object.
        """
        return self._mobsim

    @mobsim.setter
    def mobsim(self, mobsims: list[MobilitySim]):
        """Set the mobility simulation object.

        Args:
            mobsims (list[MobilitySim]): A list of MobilitySim objects.

        Raises:
            ValueError: If the provided objects are not of type MobilitySim or if trip generation/distribution has not been performed.
        """
        for mobsim_n in mobsims:
            if not isinstance(mobsim_n, MobilitySim):
                print(f"ERROR \t A MobilitySim object is required as an input.")
                return

            # Check if trip generation has been performed
            if not 'n_outflows' in mobsim_n.traffic_zones.columns:
                print(f"ERROR \t MobilitySim object - Trip Generation has not been performed.")
                return

            # Check if trip distribution has been performed
            if not 'Origin' in mobsim_n.flows.columns:
                print(f"ERROR \t MobilitySim object - Trip Distribution has not been performed.")
                return

        self._mobsim = mobsims

    # TAZ Properties
    @property
    def taz_properties(self) -> pd.DataFrame:
        """Get the properties of Traffic Analysis Zones (TAZ).

        Returns:
            pd.DataFrame: The TAZ properties DataFrame.
        """
        return self._taz_properties

    @taz_properties.setter
    def taz_properties(self, mobsim: list[MobilitySim]):
        """Set the properties of Traffic Analysis Zones (TAZ) by summing the relevant columns.

        Args:
            mobsim (list[MobilitySim]): A list of MobilitySim objects to extract TAZ properties.
        """
        # Create a list of TAZ properties
        df_list = []
        for mobsim_n in mobsim:
            df_list.append(mobsim_n.traffic_zones)

        # List of columns to sum
        columns_to_sum = ['n_outflows', 'n_inflows', 'fkt_outflows', 'fkt_inflows', 'vkt_outflows', 'vkt_inflows']

        # Initialize a DataFrame by summing the columns across all DataFrames in the list
        summed_df = pd.concat([df[columns_to_sum] for df in df_list]).groupby(level=0).sum()

        # Add back the non-summed columns from the first DataFrame (e.g., 'id')
        result_df = df_list[0][['id', 'geometric_center', 'bbox', 'is_within_target_area', 'intermediate_stops']].copy()
        result_df[columns_to_sum] = summed_df

        self._taz_properties = result_df

    # EV Fleet dictionnary
    @property
    def ev_fleet(self) -> dict:
        """Get the electric vehicle fleet.

        Returns:
            dict: The electric vehicle fleet dictionary.
        """
        return self._ev_fleet

    @ev_fleet.setter
    def ev_fleet(self, ev_fleet_value: dict):
        """Set the electric vehicle fleet.

        Args:
            ev_fleet_value (dict): A dictionary representing the electric vehicle fleet.
        """
        self._ev_fleet = ev_fleet_value

    # Charging Efficiency
    @property
    def charging_efficiency(self) -> float:
        """Get the charging efficiency.

        Returns:
            float: The charging efficiency (between 0 and 1).
        """
        return self._charging_efficiency

    @charging_efficiency.setter
    def charging_efficiency(self, charging_efficiency_value: float):
        """Set the charging efficiency.

        Args:
            charging_efficiency_value (float): The charging efficiency value (should be between 0 and 1).

        Raises:
            ValueError: If the charging efficiency value is not between 0 and 1.
        """
        if charging_efficiency_value < 0.0 or charging_efficiency_value > 1.0:
            raise ValueError("The EV charging efficiency should be between 0 and 1")

        self._charging_efficiency = charging_efficiency_value

    # Time step
    @property
    def time_step(self) -> int:
        """Get the time step.

        Returns:
            int: The time step value.
        """
        return self._time_step

    @time_step.setter
    def time_step(self, time_step_value: int):
        """Set the time step.

        Args:
            time_step_value (int): The time step value.
        """
        self._time_step = time_step_value

    # Scenario definition
    @property
    def scenario_definition(self) -> dict:
        """Get the scenario definition.

        Returns:
            dict: The scenario definition dictionary.
        """
        return self._scenario_definition

    @scenario_definition.setter
    def scenario_definition(self, cs: dict):
        """Set the scenario definition.

        Args:
            cs (dict): A dictionary containing scenario definition parameters.

        Raises:
            ValueError: If the scenario shares or arrival times are not within valid ranges.
        """
        if cs['Origin']['Share'] > 1.0 or cs['Origin']['Share'] < 0 or cs['Destination']['Share'] > 1.0 or cs['Destination']['Share'] < 0 or cs['Intermediate']['Share'] > 1.0 or cs['Intermediate']['Share'] < 0:
            raise ValueError("Share of charging at origin, destination, or intermediate stops should be between 0 and 1")

        if (cs['Origin']['Share'] + cs['Destination']['Share'] + cs['Intermediate']['Share']) != 1.0:
            raise ValueError("The total of the shares at the origin, destination and intermediate stops does not sum up to 1")

        self._scenario_definition = cs

    # Charging Demand
    @property
    def charging_demand(self) -> pd.DataFrame:
        """Get the charging demand DataFrame.

        Returns:
            pd.DataFrame: The charging demand DataFrame.
        """
        return self._charging_demand

    @charging_demand.setter
    def charging_demand(self, charging_demand_df: pd.DataFrame):
        """Set the charging demand DataFrame.

        Args:
            charging_demand_df (pd.DataFrame): A DataFrame containing charging demand data.
        """
        self._charging_demand = charging_demand_df

    # Charging Profile
    @property
    def charging_profile(self) -> pd.DataFrame:
        """Get the charging profile DataFrame.

        Returns:
            pd.DataFrame: The charging profile DataFrame.
        """
        return self._charging_profile

    @charging_profile.setter
    def charging_profile(self, charging_profile_df: pd.DataFrame):
        """Set the charging profile DataFrame.

        Args:
            charging_profile_df (pd.DataFrame): A DataFrame containing charging profile data.
        """
        self._charging_profile = charging_profile_df

    #######################################
    ####### Spatial Charging Demand #######
    #######################################

    def spatial_charging_demand(self) -> None:
        """Compute the spatial charging demand and update the charging_demand DataFrame.

        This method calculates the charging demand at each Traffic Analysis Zone (TAZ)
        based on the scenario definition and the properties of the TAZs. It updates the
        charging_demand attribute with the computed results.

        Returns:
            None: This method does not return a value; it updates the internal charging_demand DataFrame.
        """
        print(f"INFO \t COMPUTING THE SPATIAL CHARGING DEMAND")

        # Inputs
        share_origin = self.scenario_definition['Origin']['Share']
        share_destination = self.scenario_definition['Destination']['Share']
        share_intermediate = self.scenario_definition['Intermediate']['Share']

         # Compute the vehicle share - weighted ev consumption 
        average_ev_consumption = 0
        for vehicle in self.ev_fleet:
            average_ev_consumption += vehicle[0]['ev_consumption'] * vehicle[1]

        # Dispatch the remaining charging needs on intermediate stops assuming the same average energy needs per car
        demand_intermediate = 2 * self.taz_properties['fkt_inflows'].sum() * average_ev_consumption / self.charging_efficiency * share_intermediate
        vehicles_intermediate = self.taz_properties['n_inflows'].sum() * share_intermediate
        demand_per_vehicle = (demand_intermediate / vehicles_intermediate) if demand_intermediate > 0 else 0
        total_intermediate_stops = self.taz_properties['intermediate_stops'].sum()

        data = []

        # Iterate over TAZs to get the charging for vehicles charging at origin and at destination
        for index, row in self.taz_properties.iterrows():
            # Get TAZ id and bbox
            taz_id = row['id']
            bbox = row['bbox']
            is_within_target_area = row['is_within_target_area']

            # Number of vehicles charging at origin and destination
            vehicles_origin = int(round((row['n_outflows'] * share_origin)))
            vehicles_destination = int(round((row['n_inflows'] * share_destination)))

            # Total charging demand           

            Etot_origin = (2 * row['fkt_outflows'] * share_origin * average_ev_consumption /
                           self.charging_efficiency)  # Multiply by 2 (origin-destination-origin)
            Etot_destination = (2 * row['fkt_inflows'] * share_destination * average_ev_consumption /
                                self.charging_efficiency)  # Multiply by 2 (origin-destination-origin)

            # For the intermediate stops (simplified approach)
            Etot_intermediate = demand_intermediate * (row['intermediate_stops'] / total_intermediate_stops)
            vehicles_intermediate = round(Etot_intermediate / demand_per_vehicle) if Etot_intermediate > 0 else 0

            # Average charging demand per vehicle

            E0_origin = (Etot_origin / vehicles_origin) if vehicles_origin > 0 else 0
            E0_destination = (Etot_destination / vehicles_destination) if vehicles_destination > 0 else 0
            E0_intermediate = demand_per_vehicle if vehicles_intermediate > 0 else 0

            data.append({'id': taz_id,
                          'bbox': bbox,
                          'is_within_target_area': is_within_target_area,
                          'n_vehicles_origin': vehicles_origin,
                          'n_vehicles_destination': vehicles_destination,
                          'n_vehicles_intermediate': vehicles_intermediate,
                          'E0_origin_kWh': E0_origin,
                          'E0_destination_kWh': E0_destination,
                          'E0_intermediate_kWh': E0_intermediate,                          
                          'Etot_origin_kWh': Etot_origin,
                          'Etot_destination_kWh': Etot_destination,
                          'Etot_intermediate_kWh': Etot_intermediate})

        df = pd.DataFrame(data)

        self._charging_demand = df

        print(f" \t Charging demand. At origin: {self.charging_demand['Etot_origin_kWh'].sum()} kWh - At destination: {self.charging_demand['Etot_destination_kWh'].sum()} kWh - At intermediate stops: {self.charging_demand['Etot_intermediate_kWh'].sum()} kWh")
        print(f" \t Vehicles charging. At origin: {self.charging_demand['n_vehicles_origin'].sum()} - At destination: {self.charging_demand['n_vehicles_destination'].sum()} - At intermediate stops: {self.charging_demand['n_vehicles_intermediate'].sum()}")
        
    @property
    def charging_demand(self) -> pd.DataFrame:
        """Get the charging demand DataFrame.

        Returns:
            pd.DataFrame: The charging demand DataFrame.
        """
        return self._charging_demand

    #######################################
    ###### Temporal Charging Demand #######
    #######################################

    def temporal_charging_demand(self) -> None:
        """
        Computes the temporal charging demand (charging curve) based on vehicle charging profiles 
        at origin and destination, and stores the results in a DataFrame.

        Returns:
            None
        """
        print("INFO \t COMPUTING THE TEMPORAL CHARGING DEMAND (CHARGING CURVE)")

        travel_time_origin_destination_hours = self.scenario_definition['Travel time origin-destination']

        time_origin, power_profile_origin, num_cars_charging_origin = self.eval_charging_profile(
            origin_or_destination="Origin", 
            travel_time_origin_destination_hours=travel_time_origin_destination_hours
        )
        time_destination, power_profile_destination, num_cars_charging_destination = self.eval_charging_profile(
            origin_or_destination="Destination", 
            travel_time_origin_destination_hours=travel_time_origin_destination_hours
        )
        time_intermediate, power_profile_intermediate, num_cars_charging_intermediate = self.eval_charging_profile(
            origin_or_destination="Intermediate", 
            travel_time_origin_destination_hours=travel_time_origin_destination_hours
        )

        print(f" \t Max. number of vehicles charging simultaneously. "
              f"At origin: {np.max(num_cars_charging_origin)} - At destination: {np.max(num_cars_charging_destination)} - At intermediate stops: {np.max(num_cars_charging_intermediate)}")
        print(f" \t Peak power. At origin: {np.max(power_profile_origin)} MW - At destination: {np.max(power_profile_destination)} MW - At intermediate: {np.max(power_profile_intermediate)} MW")

        # Create DataFrames for each time series
        df = pd.DataFrame({
            'Time': time_origin,
            'Charging profile at origin (MW)': power_profile_origin,
            'Number of cars charging at origin': num_cars_charging_origin,
            'Charging profile at destination (MW)': power_profile_destination,
            'Number of cars charging at destination': num_cars_charging_destination,
            'Charging profile at intermediate stops (MW)': power_profile_intermediate,
            'Number of cars charging at intermediate stops': num_cars_charging_intermediate,
            'Total (MW)': power_profile_origin + power_profile_destination + power_profile_intermediate
        })

        self._charging_profile = df

    @property
    def charging_profile(self) -> pd.DataFrame:
        """
        Returns the charging profile DataFrame containing the temporal charging demand data.

        Returns:
            pd.DataFrame: DataFrame with the charging profile data.
        """
        return self._charging_profile

    def eval_charging_profile(self, origin_or_destination: str = "Origin", travel_time_origin_destination_hours: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluates the charging profile at either the origin or destination based on arrival and 
        departure times, vehicle counts, and smart charging percentages.

        Args:
            origin_or_destination (str): Specifies whether the evaluation is for the "Origin" 
                                          or "Destination".
            travel_time_origin_destination_hours (float): Travel time in hours from origin to 
                                                          destination.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple containing:
                - time (np.ndarray): Array of time points.
                - power_demand (np.ndarray): Array of power demands in MW.
                - num_cars_charging (np.ndarray): Array of numbers of cars charging.
        """

        """
        Inform if charging at origin or destination and get corresponding parameters 
        """
        if origin_or_destination == "Origin":            
            arrival_time = self.scenario_definition['Origin']['Arrival time']
            departure_time = self.scenario_definition['Destination']['Arrival time']
            vehicle_counts = self.charging_demand['n_vehicles_origin'].sum()
            share_smart_charging = self.scenario_definition['Origin']['Smart charging']

            print(f"INFO \t ... Charging at origin with {share_smart_charging*100}% of smart charging...")
        elif origin_or_destination == "Destination":            
            arrival_time  = self.scenario_definition['Destination']['Arrival time']
            departure_time = self.scenario_definition['Origin']['Arrival time']
            vehicle_counts = self.charging_demand['n_vehicles_destination'].sum()
            share_smart_charging = self.scenario_definition['Destination']['Smart charging']

            print(f"INFO \t ... Charging at destination with {share_smart_charging*100}% of smart charging...")
        elif origin_or_destination == "Intermediate":            
            arrival_time  = self.scenario_definition['Destination']['Arrival time']
            departure_time = self.scenario_definition['Origin']['Arrival time']
            vehicle_counts = self.charging_demand['n_vehicles_intermediate'].sum()
            share_smart_charging = self.scenario_definition['Intermediate']['Smart charging']

            print(f"INFO \t ... Charging at intermediate stops with {share_smart_charging*100}% of smart charging...")
        else:
            raise ValueError("Charging should be at origin or destination or intermediate stops")

        """
        Initialize output variables
        """
        # Final time series
        time = np.arange(0, 24, self.time_step)  # time array with specified intervals
        power_demand = np.zeros_like(time)
        num_cars_charging = np.zeros_like(time)

        # Daily charging demand, max charging power, battery capacity and average days between charges for all vehicles
        charging_demands = np.zeros(vehicle_counts)
        charging_powers = np.zeros(vehicle_counts)
        battery_capacities = np.zeros(vehicle_counts)
        days_between_charges = np.zeros(len(charging_demands))

        """
        Aggregate mobility flows from all sources
        """
        # Start by copying flows from the first mobility simulation
        df_sum = self.mobsim[0].flows.copy()

        # Sum the flows from all remaining mobility simulations
        for mobsim in self.mobsim[1:]:
            df_sum = df_sum.add(mobsim.flows, fill_value=0)

        # Group by "Travel Distance (km)" and aggregate the total "Flow" for each distance
        grouped_df = df_sum.groupby('Travel Distance (km)').agg({'Flow': 'sum'}).reset_index()

        # Calculate the total flow across all distances and the associated probabilities
        total_flow = grouped_df['Flow'].sum()
        grouped_df['Probability'] = grouped_df['Flow'] / total_flow

        # Extract distances and their associated probabilities for random sampling
        distances = grouped_df['Travel Distance (km)'].values
        distance_probabilities = grouped_df['Probability'].values

        """
        Assign vehicle type and randomly assign the charging demand and maximum charging power
        In addition, we also use the battery capacity to calculate the average days between two charges (see Pareschi et al., Applied Energy, 2020)
        From this we determine if today is a charging day and keep only the vehicles that are charging
        """
        # Assign probabilities for selecting each vehicle type in the fleet
        fleet_probabilities = np.array([item[1] for item in self.ev_fleet])

        # Randomly select a vehicle type for each vehicle (based on fleet composition)
        # 'vehicle_indices' contains the index of the selected vehicle type for each charging event
        vehicle_indices = np.random.choice(len(self.ev_fleet), size=len(charging_demands), p=fleet_probabilities)

        # Loop over each vehicle and assign charging demand, power, and battery capacity based on its type
        for idx, vehicle_index in enumerate(vehicle_indices):
            vehicle = self.ev_fleet[vehicle_index]

            # Randomly assign the charging demand for the selected vehicle
            charging_demands[idx] = 2 * np.random.choice(distances, p=distance_probabilities) * vehicle[0]['ev_consumption'] / self.charging_efficiency

            # Randomly assign the charging power based on whether the charging is done at Origin or Destination
            if origin_or_destination == "Origin":
                available_charging_power = [item[0] for item in vehicle[0]['charger_power']['Origin']]
                probabilities = [item[1] for item in vehicle[0]['charger_power']['Origin']]
            elif origin_or_destination == "Destination":
                available_charging_power = [item[0] for item in vehicle[0]['charger_power']['Destination']]
                probabilities = [item[1] for item in vehicle[0]['charger_power']['Destination']]
            elif origin_or_destination == "Intermediate":
                available_charging_power = [item[0] for item in vehicle[0]['charger_power']['Intermediate']]
                probabilities = [item[1] for item in vehicle[0]['charger_power']['Intermediate']]

            # Randomly select a charging power
            charging_powers[idx] = np.random.choice(available_charging_power, p=probabilities)

            # Assign the reduced battery capacity for the selected vehicle
            battery_capacities[idx] = vehicle[0]['battery_capacity'] 

            # Calculate the average number of days between charges for the current vehicle
            # Apply the function calculate_days_between_charges_single_vehicle to each vehicle
            # Based on Pareschi et al., Applied Energy, 2020
            days_between_charges[idx] = hlp.calculate_days_between_charges_single_vehicle(
                daily_charging_demand=charging_demands[idx], 
                battery_capacity=battery_capacities[idx]
            )

            # Determine if today is a charging day for the current vehicle
            charging_probability = 1 / days_between_charges[idx]

            # Generate a random number to determine if the vehicle charges today
            if np.random.rand() <= charging_probability:
                # Vehicle is charging today, modify the charging demand and power
                charging_demands[idx] *= days_between_charges[idx]  # Adjust demand for charging day
            else:
                # Vehicle is NOT charging today, set its demand and power to zero
                charging_demands[idx] = 0
                charging_powers[idx] = 0

        # Now filter out the non-charging vehicles (where demand is zero) but keep the same variable names
        charging_mask = charging_demands > 0

        # Create new arrays (overwriting the original ones) for only the charging vehicles
        charging_demands = charging_demands[charging_mask]
        charging_powers = charging_powers[charging_mask]
        battery_capacities = battery_capacities[charging_mask]
        days_between_charges = days_between_charges[charging_mask]

        print(f"INFO \t {len(charging_demands)} vehicles selected for charging out of {vehicle_counts }")

        # Update the number of vehicles to simulate
        vehicle_counts = len(charging_demands)

        """
        Assign arrival time to each vehicle using lognormal distribution
        """
        # Lognormal parameters from arrival time average and standard deviation
        # Calculate mu and sigma for the lognormal distribution
        mean_arrival = arrival_time[0]
        stddev_arrival = arrival_time[1]

        mean_departure = departure_time[0]
        stddev_departure = departure_time[1]

        if origin_or_destination == "Intermediate":        
            # For the end time, we use the charging duration + arrival time (see below), as we expect the user to stop only if he can charge without smart charging
            
            # Adding randomness to mean arrival and mean departure times
            random_mean_arrival = np.random.normal(mean_arrival, stddev_arrival/2, vehicle_counts)
            random_mean_departure = np.random.normal(mean_departure, stddev_departure/2, vehicle_counts)

            # Generate random uniform arrival times between randomized mean_arrival and mean_departure
            arrival_times = np.random.uniform(random_mean_arrival, random_mean_departure)

            # Ensure arrival times stay within 24-hour bounds (optional)
            arrival_times = arrival_times % 24

        else:
            arrival_times = np.random.normal(mean_arrival, stddev_arrival, vehicle_counts) % 24
            departure_times = np.random.normal(mean_departure, stddev_departure - travel_time_origin_destination_hours, vehicle_counts) % 24

        """
        Assign vehicles with smart charging
        """
        # Number of vehicles with smart charging
        num_true = int(len(charging_demands) * share_smart_charging)

        # Create an array with num_true True values and the rest False
        vehicles_with_smartcharging = np.array([True] * num_true + [False] * (len(charging_demands) - num_true))

        # Shuffle the array to randomize the positions of True and False
        np.random.shuffle(vehicles_with_smartcharging)

        """
        Compute the aggregated charging demand by looping over all vehicles
        """
        # Compute start and end indices for charging for each vehicle
        # Wrap around to fit within 24 hours
        start_indices = np.searchsorted(time, arrival_times)
        charging_durations = charging_demands / charging_powers
        end_times = (arrival_times + charging_durations) % 24

        if origin_or_destination == "Intermediate":  
            departure_times = end_times

        end_indices = np.searchsorted(time, end_times)

        # Some preleminary checks
        if np.any(charging_durations <= self.time_step):
            print("ALERT \t Some charging duration are smaller than the timestep. This may lead to inaccurate result.")

        if np.any(charging_durations > 12):
            print("ALERT \t Some charging duration are greater than 12 hours. This might not be consistent with departure/arrival times.")

        # Create masks for the ranges where the power demand should be applied
        mask_wrap_around = end_indices < start_indices
        mask_no_wrap = ~mask_wrap_around

        # Apply power demand for no-wrap cases
        for i in np.where(mask_no_wrap)[0]:
            if vehicles_with_smartcharging[i]:
                continue
            power_demand[start_indices[i]:end_indices[i]] += charging_powers[i]
            num_cars_charging[start_indices[i]:end_indices[i]] += 1

        # Apply power demand for wrap-around cases
        for i in np.where(mask_wrap_around)[0]:
            if vehicles_with_smartcharging[i]:
                continue
            power_demand[start_indices[i]:] += charging_powers[i]
            num_cars_charging[start_indices[i]:] += 1
            power_demand[:end_indices[i]] += charging_powers[i]
            num_cars_charging[:end_indices[i]] += 1

        # Convert power demand to MWh
        power_demand_mw = power_demand / 1000  # converting kW to MW

        """
        Smart charging 
        """
        peak_power = 0  # Initialise le pic total de puissance à zéro
        power_demand = np.zeros(len(time))
        num_cars_charging_smart = np.zeros(len(time))
        
        for i in range(len(arrival_times)):
            if not vehicles_with_smartcharging[i]:
                continue

            # Calcul de l'intervalle de chargement
            start_idx = np.searchsorted(time, arrival_times[i])
            end_idx = np.searchsorted(time, departure_times[i])
            if end_idx < start_idx:
                end_idx += len(time) # Gérer le wrap-around pour une journée de 24 heures
        
            # Calcul initial de la demande de charge
            remaining_demand = charging_demands[i]
            max_power = charging_powers[i]

            # Temporary array to track cars charging during uniform distribution
            temp_num_cars_charging = np.zeros(len(time))
            
            # Parcourir l'intervalle de temps pour minimiser le pic de puissance
            for t in range(start_idx, end_idx):
                current_time_idx = t % len(time)  # Boucle sur 24 heures
                current_total_power = power_demand[current_time_idx]
                
                # Si la puissance actuelle est inférieure au pic total, charger au maximum permis
                if current_total_power < peak_power:
                    charge_power = min(peak_power - current_total_power, max_power)
                    charge_energy = charge_power * self.time_step

                    # S'assurer de ne pas charger plus que la demande restante
                    if charge_energy > remaining_demand:
                        charge_energy = remaining_demand
                        charge_power = charge_energy / self.time_step  # Ajuster la puissance en fonction de l'énergie restante

                    power_demand[current_time_idx] += charge_power 
                    remaining_demand -= charge_energy
                    num_cars_charging_smart[current_time_idx] += 1  # Increment number of cars charging
                    
                    # Si toute l'énergie est chargée, passer au véhicule suivant
                    if remaining_demand <= 0:
                        break
            
            # Si de l'énergie reste à charger, répartir uniformément
            if remaining_demand > 0:
                # Manage the case where the departure time is smaller than the arrival time
                if departure_times[i] < arrival_times[i]:
                    # Charging spans over midnight
                    charging_duration = (24 - arrival_times[i]) + departure_times[i]
                else:
                    # No wrap-around
                    charging_duration = departure_times[i] - arrival_times[i]

                charge_power_uniform = remaining_demand / charging_duration

                for t in range(start_idx, end_idx):
                    current_time_idx = t % len(time)
                    power_demand[current_time_idx] += charge_power_uniform
                    temp_num_cars_charging[current_time_idx] += 1  # Increment temp count for cars charging
        
            # Mettre à jour le pic total de puissance si nécessaire
            peak_power = np.max(power_demand)

            # Combine temp_num_cars_charging with num_cars_charging
            num_cars_charging_smart = np.maximum(num_cars_charging_smart, temp_num_cars_charging)

        power_demand_mw = power_demand_mw + (power_demand / 1000)  # Convertir de kW en MW
        num_cars_charging = num_cars_charging + num_cars_charging_smart

        return time, power_demand_mw, num_cars_charging

    #######################################
    ### Post-processing & visualisation ###
    #######################################

    def chargingdemand_total_to_map(self) -> folium.Map:
        """
        Creates a Folium map visualizing the total charging demand at both origin and destination 
        points based on the charging demand data.

        The map includes:
        - Administrative boundaries
        - TAZ boundaries represented as rectangles
        - Charging demand at origin displayed with a color scale
        - Charging demand at destination displayed with a color scale
        - A legend for the color scale
        - Layer control for toggling the layers on the map

        Returns:
            folium.Map: The generated Folium map object.
        """
        df = self.charging_demand

        # 1. Create an empty map

        m = folium.Map(location=self.mobsim[0].centroid_coords, zoom_start=12, tiles='CartoDB Positron', control_scale=True) # Create the map

        # 2. Add Administrative Boundaries
        def style_function(feature):
            return {
                'color': 'blue',
                'weight': 3,
                'fillColor': 'none',
            }
        
        folium.GeoJson(self.mobsim[0].target_area, name='Administrative boundary', style_function=style_function).add_to(m)

        # 2. Add TAZ boundaries

        # Function to add rectangles to the map
        def add_rectangle(row):
            # Parse the WKT string to create a Polygon object
            bbox_polygon = row['bbox']
            bbox_coords = bbox_polygon.bounds
            
            # Add rectangle to map
            folium.Rectangle(
                bounds=[(bbox_coords[1], bbox_coords[0]), (bbox_coords[3], bbox_coords[2])],
                color='grey',
                fill=True,
                fill_color='grey',
                fill_opacity=0.0
            ).add_to(m)

        # Apply the function to each row in the DataFrame
        df.apply(add_rectangle, axis=1)

        # 3. Charging AT ORIGIN

        # Normalize data for color scaling
        linear = cm.LinearColormap(["#edf8b1", "#7fcdbb", "#2c7fb8"], vmin=0, vmax=max(df['Etot_origin_kWh'].max(), df['Etot_destination_kWh'].max()) )

        # Create a feature group for all polygons
        feature_group = folium.FeatureGroup(name='Charging demand at Origin')

        # Add polygons to the feature group
        for idx, row in df.iterrows():
            bbox_polygon = row['bbox']
            bbox_coords = bbox_polygon.bounds

            # Create a rectangle for each row
            rectangle = folium.Rectangle(
                bounds=[(bbox_coords[1], bbox_coords[0]), (bbox_coords[3], bbox_coords[2])],
                color=None,
                fill=True,
                fill_color=linear(row['Etot_origin_kWh']),
                fill_opacity=0.7
                #popup=f"ID: {row['id']} - Trips: {int(row['n_outflows'])}"
            )

            # Add the rectangle to the feature group
            rectangle.add_to(feature_group)

        # Add the feature group to the map
        feature_group.add_to(m)

        # 4. Charging AT DESTINATION

        # Create a feature group for all polygons
        feature_group = folium.FeatureGroup(name='Charging demand at Destination')

        # Add polygons to the feature group
        for idx, row in df.iterrows():
            bbox_polygon = row['bbox']
            bbox_coords = bbox_polygon.bounds

            # Create a rectangle for each row
            rectangle = folium.Rectangle(
                bounds=[(bbox_coords[1], bbox_coords[0]), (bbox_coords[3], bbox_coords[2])],
                color=None,
                fill=True,
                fill_color=linear(row['Etot_destination_kWh']),
                fill_opacity=0.7
                #popup=f"ID: {row['id']} - Trips: {int(row['n_outflows'])}"
            )

            # Add the rectangle to the feature group
            rectangle.add_to(feature_group)

        # Add the feature group to the map
        feature_group.add_to(m)

        # Add the color scale legend to the map
        linear.caption = 'Charging demand (kWh)'
        linear.add_to(m)

        # 5. Charging AT INTERMEDIATE STOPS

        # Create a feature group for all polygons
        feature_group = folium.FeatureGroup(name='Charging demand at Intermediate stops')

        # Add polygons to the feature group
        for idx, row in df.iterrows():
            bbox_polygon = row['bbox']
            bbox_coords = bbox_polygon.bounds

            # Create a rectangle for each row
            rectangle = folium.Rectangle(
                bounds=[(bbox_coords[1], bbox_coords[0]), (bbox_coords[3], bbox_coords[2])],
                color=None,
                fill=True,
                fill_color=linear(row['Etot_intermediate_kWh']),
                fill_opacity=0.7
                #popup=f"ID: {row['id']} - Trips: {int(row['n_outflows'])}"
            )

            # Add the rectangle to the feature group
            rectangle.add_to(feature_group)

        # Add the feature group to the map
        feature_group.add_to(m)

        # Add the color scale legend to the map
        linear.caption = 'Charging demand (kWh)'
        linear.add_to(m)

        # Add Layer Control and Save 

        folium.LayerControl().add_to(m)

        return m

    def chargingdemand_pervehicle_to_map(self) -> folium.Map:
        """
        Creates a Folium map visualizing the charging demand per vehicle at both origin and destination points 
        based on the charging demand data.

        The map includes:
        - Administrative boundaries
        - TAZ boundaries represented as rectangles
        - Charging needs per vehicle at origin displayed with a color scale
        - Charging needs per vehicle at destination displayed with a color scale
        - A legend for the color scale
        - Layer control for toggling the layers on the map

        Returns:
            folium.Map: The generated Folium map object.
        """
        df = self.charging_demand

        # 1. Create an empty map

        m = folium.Map(location=self.mobsim[0].centroid_coords, zoom_start=12, tiles='CartoDB Positron', control_scale=True) # Create the map

        # 2. Add Administrative Boundaries
        def style_function(feature):
            return {
                'color': 'blue',
                'weight': 3,
                'fillColor': 'none',
            }
        
        folium.GeoJson(self.mobsim[0].target_area, name='Administrative boundary', style_function=style_function).add_to(m)

        # 2. Add TAZ boundaries

        # Function to add rectangles to the map
        def add_rectangle(row):
            # Parse the WKT string to create a Polygon object
            bbox_polygon = row['bbox']
            bbox_coords = bbox_polygon.bounds
            
            # Add rectangle to map
            folium.Rectangle(
                bounds=[(bbox_coords[1], bbox_coords[0]), (bbox_coords[3], bbox_coords[2])],
                color='grey',
                fill=True,
                fill_color='grey',
                fill_opacity=0.0
            ).add_to(m)

        # Apply the function to each row in the DataFrame
        df.apply(add_rectangle, axis=1)

        # 3. Charging AT ORIGIN

        # Normalize data for color scaling
        linear = cm.LinearColormap(["#ffeda0", "#feb24c", "#f03b20"], vmin=0, vmax=max(df['E0_origin_kWh'].max(),df['E0_destination_kWh'].max()) )

        # Create a feature group for all polygons
        feature_group = folium.FeatureGroup(name='Charging need per vehicle at Origin')

        # Add polygons to the feature group
        for idx, row in df.iterrows():
            bbox_polygon = row['bbox']
            bbox_coords = bbox_polygon.bounds

            # Create a rectangle for each row
            rectangle = folium.Rectangle(
                bounds=[(bbox_coords[1], bbox_coords[0]), (bbox_coords[3], bbox_coords[2])],
                color=None,
                fill=True,
                fill_color=linear(row['E0_origin_kWh']),
                fill_opacity=0.7
                #popup=f"ID: {row['id']} - Trips: {int(row['n_outflows'])}"
            )

            # Add the rectangle to the feature group
            rectangle.add_to(feature_group)

        # Add the feature group to the map
        feature_group.add_to(m)

        # 4. Charging AT DESTINATION

        # Create a feature group for all polygons
        feature_group = folium.FeatureGroup(name='Charging need per vehicle at Destination')

        # Add polygons to the feature group
        for idx, row in df.iterrows():
            bbox_polygon = row['bbox']
            bbox_coords = bbox_polygon.bounds

            # Create a rectangle for each row
            rectangle = folium.Rectangle(
                bounds=[(bbox_coords[1], bbox_coords[0]), (bbox_coords[3], bbox_coords[2])],
                color=None,
                fill=True,
                fill_color=linear(row['E0_destination_kWh']),
                fill_opacity=0.7
                #popup=f"ID: {row['id']} - Trips: {int(row['n_outflows'])}"
            )

            # Add the rectangle to the feature group
            rectangle.add_to(feature_group)

        # Add the feature group to the map
        feature_group.add_to(m)

        # 5. Charging AT INTERMEDIATE STOPS

        # Create a feature group for all polygons
        feature_group = folium.FeatureGroup(name='Charging need per vehicle at Intermediate stops')

        # Add polygons to the feature group
        for idx, row in df.iterrows():
            bbox_polygon = row['bbox']
            bbox_coords = bbox_polygon.bounds

            # Create a rectangle for each row
            rectangle = folium.Rectangle(
                bounds=[(bbox_coords[1], bbox_coords[0]), (bbox_coords[3], bbox_coords[2])],
                color=None,
                fill=True,
                fill_color=linear(row['E0_intermediate_kWh']),
                fill_opacity=0.7
                #popup=f"ID: {row['id']} - Trips: {int(row['n_outflows'])}"
            )

            # Add the rectangle to the feature group
            rectangle.add_to(feature_group)

        # Add the feature group to the map
        feature_group.add_to(m)

        # Add the color scale legend to the map
        linear.caption = 'Charging needs (kWh/car)'
        linear.add_to(m)

        # Add Layer Control and Save 

        folium.LayerControl().add_to(m)

        return m

    def chargingdemand_nvehicles_to_map(self) -> folium.Map:
        """
        Creates a Folium map visualizing the number of vehicles charging at both origin and destination points 
        based on the charging demand data.

        The map includes:
        - Administrative boundaries
        - TAZ boundaries represented as rectangles
        - Number of vehicles charging at origin displayed with a color scale
        - Number of vehicles charging at destination displayed with a color scale
        - A legend for the color scale
        - Layer control for toggling the layers on the map

        Returns:
            folium.Map: The generated Folium map object.
        """
        df = self.charging_demand

        # 1. Create an empty map

        m = folium.Map(location=self.mobsim[0].centroid_coords, zoom_start=12, tiles='CartoDB Positron', control_scale=True) # Create the map

        # 2. Add Administrative Boundaries
        def style_function(feature):
            return {
                'color': 'blue',
                'weight': 3,
                'fillColor': 'none',
            }
        
        folium.GeoJson(self.mobsim[0].target_area, name='Administrative boundary', style_function=style_function).add_to(m)

        # 2. Add TAZ boundaries

        # Function to add rectangles to the map
        def add_rectangle(row):
            # Parse the WKT string to create a Polygon object
            bbox_polygon = row['bbox']
            bbox_coords = bbox_polygon.bounds
            
            # Add rectangle to map
            folium.Rectangle(
                bounds=[(bbox_coords[1], bbox_coords[0]), (bbox_coords[3], bbox_coords[2])],
                color='grey',
                fill=True,
                fill_color='grey',
                fill_opacity=0.0
            ).add_to(m)

        # Apply the function to each row in the DataFrame
        df.apply(add_rectangle, axis=1)

        # 3. Charging AT ORIGIN

        # Normalize data for color scaling
        linear = cm.LinearColormap(["#e0ecf4", "#9ebcda", "#8856a7"], vmin=0, vmax=max(df['n_vehicles_origin'].max(), df['n_vehicles_destination'].max()) )

        # Create a feature group for all polygons
        feature_group = folium.FeatureGroup(name='Number of vehicles charging at Origin')

        # Add polygons to the feature group
        for idx, row in df.iterrows():
            bbox_polygon = row['bbox']
            bbox_coords = bbox_polygon.bounds

            # Create a rectangle for each row
            rectangle = folium.Rectangle(
                bounds=[(bbox_coords[1], bbox_coords[0]), (bbox_coords[3], bbox_coords[2])],
                color=None,
                fill=True,
                fill_color=linear(row['n_vehicles_origin']),
                fill_opacity=0.7
                #popup=f"ID: {row['id']} - Trips: {int(row['n_outflows'])}"
            )

            # Add the rectangle to the feature group
            rectangle.add_to(feature_group)

        # Add the feature group to the map
        feature_group.add_to(m)

        # 4. Charging AT DESTINATION

        # Create a feature group for all polygons
        feature_group = folium.FeatureGroup(name='Number of vehicles charging at Destination')

        # Add polygons to the feature group
        for idx, row in df.iterrows():
            bbox_polygon = row['bbox']
            bbox_coords = bbox_polygon.bounds

            # Create a rectangle for each row
            rectangle = folium.Rectangle(
                bounds=[(bbox_coords[1], bbox_coords[0]), (bbox_coords[3], bbox_coords[2])],
                color=None,
                fill=True,
                fill_color=linear(row['n_vehicles_destination']),
                fill_opacity=0.7
                #popup=f"ID: {row['id']} - Trips: {int(row['n_outflows'])}"
            )

            # Add the rectangle to the feature group
            rectangle.add_to(feature_group)

        # Add the feature group to the map
        feature_group.add_to(m)

        # 4. Charging AT INTERMEDIATE STOPS

        # Create a feature group for all polygons
        feature_group = folium.FeatureGroup(name='Number of vehicles charging at Intermediate stops')

        # Add polygons to the feature group
        for idx, row in df.iterrows():
            bbox_polygon = row['bbox']
            bbox_coords = bbox_polygon.bounds

            # Create a rectangle for each row
            rectangle = folium.Rectangle(
                bounds=[(bbox_coords[1], bbox_coords[0]), (bbox_coords[3], bbox_coords[2])],
                color=None,
                fill=True,
                fill_color=linear(row['n_vehicles_intermediate']),
                fill_opacity=0.7
                #popup=f"ID: {row['id']} - Trips: {int(row['n_outflows'])}"
            )

            # Add the rectangle to the feature group
            rectangle.add_to(feature_group)

        # Add the feature group to the map
        feature_group.add_to(m)

        # Add the color scale legend to the map
        linear.caption = 'Number of vehicles'
        linear.add_to(m)

        # Add Layer Control and Save 

        folium.LayerControl().add_to(m)

        return m
