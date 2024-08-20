# coding: utf-8

""" 
ChargingDemand

A class to simulate the daily charging demand using mobility simulation and assumptions 
regarding the charging scenario and the EV fuel consumption and charging efficiency
"""

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
from dotenv import load_dotenv
import math
import matplotlib.pyplot as plt

from evpv import helpers as hlp
from evpv.mobilitysim import MobilitySim

load_dotenv() # take environment variables from .env

INPUT_PATH = Path( str(os.getenv("INPUT_PATH")) )
OUTPUT_PATH = Path( str(os.getenv("OUTPUT_PATH")) )

class ChargingScenario:

    #######################################
    ############# ATTRIBUTES ##############
    #######################################
    
    # Mobility Simulations used to feed the charging model
    mobsim = []

    # EV fuel consumption and charging efficiency
    ev_consumption = .0
    charging_efficiency = .0

    # Charging scenario definition
    charging_scenario = {}

    # Outputs
    taz_properties = pd.DataFrame() # Relevant properties from Mobility Simulation for each TAZ    
    charging_demand = pd.DataFrame() # Charging need distribution across TAZs
    charging_profile_origin = None # Charging curve for EVs 
    charging_profile_destination = None # Charging curve for EVs 

    #######################################
    ############### METHODS ###############
    #######################################
    
    ############# Constructor #############
    ####################################### 

    def __init__(self, mobsim, ev_consumption, charging_efficiency, time_step, scenario_definition):
        print("---")
        print(f"INFO \t ChargingDemand object initialisation")

        self.set_mobsim(mobsim)
        self.set_taz_properties() # Combine TAZ properties of each mobsim into a single effective TAZ properties df

        self.set_charging_efficiency(charging_efficiency)
        self.set_ev_consumption(ev_consumption)

        self.set_time_step(time_step)
        self.set_scenario_definition(scenario_definition)        

        print(f"INFO \t ChargingDemand object created")
        print(f" \t Number of trips: {self.taz_properties['n_inflows'].sum()} (n_in) // {self.taz_properties['n_outflows'].sum()} (n_out)")
        print(f" \t FKT (origin to destination): {self.taz_properties['fkt_inflows'].sum()} km")
        print(f" \t Average VKT (origin to destination): {self.taz_properties['fkt_inflows'].sum() / self.taz_properties['n_inflows'].sum()} km")

        self.charging_demand()

        print(f" \t Charging demand. At origin: {self.charging_demand['Etot_origin_kWh'].sum()} kWh - At destination: {self.charging_demand['Etot_destination_kWh'].sum()} kWh")
        print(f" \t Vehicles charging. At origin: {self.charging_demand['n_vehicles_origin'].sum()} - At destination: {self.charging_demand['n_vehicles_destination'].sum()}")

        self.charging_profile_origin()
        self.charging_profile_destination()

        print(f"INFO \t Charging profile:")
        print(f" \t Max. number of vehicles charging simultaneously. At origin: {self.charging_profile_origin[3]} - At destination: {self.charging_profile_destination[3]}")
        print(f" \t Peak power. At origin: {np.max(self.charging_profile_origin[1])} MW - At destination: {np.max(self.charging_profile_destination[1])} MW")

        print("---")

    ############# Setters #############
    ###################################

    def set_mobsim(self, mobsim):
        """ Setter for the mobsim attribute.
        """
        for mobsim_n in mobsim:
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

        self.mobsim = mobsim

    def set_vkt_offset(self, vkt_offset):
        """ Setter for the vkt_offset attribute.
        """
        self.vkt_offset = vkt_offset

    def set_ev_consumption(self, ev_consumption):
        self.ev_consumption = ev_consumption

    def set_charging_efficiency(self, charging_efficiency):
        self.charging_efficiency = charging_efficiency

    def set_taz_properties(self):
        """ Setter for the taz_properties attribute by summing the relevant properties for each mobsim
        """
        # Create a list of TAZ properties
        df_list = []
        for mobsim_n in self.mobsim:
            df_list.append(mobsim_n.traffic_zones)

        # List of columns to sum
        columns_to_sum = ['n_outflows', 'n_inflows', 'fkt_outflows', 'fkt_inflows', 'vkt_outflows', 'vkt_inflows']

        # Initialize a DataFrame by summing the columns across all DataFrames in the list
        summed_df = pd.concat([df[columns_to_sum] for df in df_list]).groupby(level=0).sum()

        # Add back the non-summed columns from the first DataFrame (e.g., 'id')
        result_df = df_list[0][['id', 'geometric_center', 'bbox', 'is_within_target_area']].copy()
        result_df[columns_to_sum] = summed_df

        self.taz_properties = result_df

    def set_time_step(self, time_step):
        """ Setter for the time_step property
        """
        self.time_step = time_step

    def set_scenario_definition(self, scenario_definition):
        """ Setter for the scenario_definition property
        """
        self.scenario_definition = scenario_definition

    ####### Charging Scenario #########
    ###################################
    def charging_demand(self):
        print(f"INFO \t Computing the charging demand distribution")

        # Inputs
        share_origin = self.scenario_definition['Origin']['Share']
        share_destination = self.scenario_definition['Destination']['Share']

        data = []

        # Iterate over TAZs
        for index, row in self.taz_properties.iterrows():
            # Get TAZ id and bbox
            taz_id = row['id']
            bbox = row['bbox']
            is_within_target_area = row['is_within_target_area']

            # Number of vehicles charging at origin and destination
            vehicles_origin = int( round((row['n_outflows']*share_origin)) )
            vehicles_destination = int( round((row['n_inflows']*share_destination)) )

            # Total charging demand 
            Etot_origin = 2 * row['fkt_outflows']*share_origin * self.ev_consumption / self.charging_efficiency # Multiply by 2 (origin-destination-origin)
            Etot_destination = 2 * row['fkt_inflows']*share_destination * self.ev_consumption / self.charging_efficiency # Multiply by 2 (origin-destination-origin)

            # Average charging demand per vehicle
            
            E0_origin = (Etot_origin / vehicles_origin) if vehicles_origin > 0 else 0
            E0_destination = (Etot_destination / vehicles_destination) if vehicles_destination > 0 else 0

            data.append({'id': taz_id,                
                'bbox': bbox, 
                'is_within_target_area': is_within_target_area, 
                'n_vehicles_origin': vehicles_origin,
                'n_vehicles_destination': vehicles_destination,
                'E0_origin_kWh': E0_origin,
                'E0_destination_kWh': E0_destination,
                'Etot_origin_kWh': Etot_origin,
                'Etot_destination_kWh': Etot_destination})

        df = pd.DataFrame(data)
        
        self.charging_demand = df

    def charging_profile_origin(self):
        """ 1. Load and Understand the Data
            2. Model Arrival Times with Lognormal Distribution
            3. Distribute Charging Demand Over Time
        """
        print(f"INFO \t Computing the aggregated charging curve for charging at origin")

        # 0. Parameters for lognormal distribution (mean and sigma)

        # Calculate sigma
        sigma = np.sqrt(np.log(1 + (self.scenario_definition['Origin']['Charging time'][1] / self.scenario_definition['Origin']['Charging time'][0])**2))
        # Calculate mu
        mu = np.log(self.scenario_definition['Origin']['Charging time'][0]) - 0.5 * sigma**2

        # 1. Load the Data

        # Expand the dataframe to list each vehicle and its charging demand
        vehicle_counts = self.charging_demand['n_vehicles_origin']
        charging_demands = self.charging_demand['E0_origin_kWh']

        # Create a list of charging demands repeated by the number of vehicles
        all_charging_demands = []
        for count, demand in zip(vehicle_counts, charging_demands):
            all_charging_demands.extend([demand] * count)

        # Convert to a numpy array
        all_charging_demands = np.array(all_charging_demands)
        num_vehicles = len(all_charging_demands)

        # 2. Model Arrival Times with Lognormal Distribution

        # Simulate arrival times (in hours) for the vehicles
        arrival_times = np.random.lognormal(mu, sigma, num_vehicles)
        arrival_times = arrival_times % 24  # Wrap around to fit within 24 hours

        # 3. Distribute Charging Demand Over Time

        # Create a time array
        time = np.arange(0, 24, self.time_step)  # time array with specified intervals
        power_demand = np.zeros_like(time)
        num_cars_plugged_in = np.zeros_like(time)

        for arrival_time, demand in zip(arrival_times, all_charging_demands):
            # Separate the values and the probabilities
            charging_powers = [item[0] for item in self.scenario_definition['Origin']['Charging power']]
            probabilities = [item[1] for item in self.scenario_definition['Origin']['Charging power']]

            # Randomly pick a value based on the specified probabilities
            charging_power = random.choices(charging_powers, weights=probabilities, k=1)[0]

            charging_duration = demand / charging_power
            start_idx = np.searchsorted(time, arrival_time)
            end_time = (arrival_time + charging_duration) % 24  # Wrap around to fit within 24 hours
            end_idx = np.searchsorted(time, end_time)

            if end_idx > start_idx:
                power_demand[start_idx:end_idx] += charging_power
                num_cars_plugged_in[start_idx:end_idx] += 1
            else:
                power_demand[start_idx:] += charging_power
                num_cars_plugged_in[start_idx:] += 1
                power_demand[:end_idx] += charging_power
                num_cars_plugged_in[:end_idx] += 1

            if charging_duration <= self.time_step:
                print("ALERT \t Charging duration is smaller than the timestep, this may lead to inaccurate results")

        # Convert power demand to MWh
        power_demand_mwh = power_demand / 1000  # converting kW to MW

        # Find the maximum number of cars plugged in at the same time
        max_cars_plugged_in = np.max(num_cars_plugged_in)
        
        self.charging_profile_origin = time, power_demand_mwh, num_cars_plugged_in, max_cars_plugged_in

    def charging_profile_destination(self):
        """ 1. Load and Understand the Data
            2. Model Arrival Times with Lognormal Distribution
            3. Distribute Charging Demand Over Time
        """
        print(f"INFO \t Computing the aggregated charging curve for charging at destination")

        # 0. Parameters for lognormal distribution (mean and sigma)

        # Calculate sigma
        sigma = np.sqrt(np.log(1 + (self.scenario_definition['Destination']['Charging time'][1] / self.scenario_definition['Destination']['Charging time'][0])**2))
        # Calculate mu
        mu = np.log(self.scenario_definition['Destination']['Charging time'][0]) - 0.5 * sigma**2

        # 1. Load and Understand the Data

        # Expand the dataframe to list each vehicle and its charging demand
        vehicle_counts = self.charging_demand['n_vehicles_destination']
        charging_demands = self.charging_demand['E0_destination_kWh']

        # Create a list of charging demands repeated by the number of vehicles
        all_charging_demands = []
        for count, demand in zip(vehicle_counts, charging_demands):
            all_charging_demands.extend([demand] * count)

        # Convert to a numpy array
        all_charging_demands = np.array(all_charging_demands)
        num_vehicles = len(all_charging_demands)

        # 2. Model Arrival Times with Lognormal Distribution

        # Simulate arrival times (in hours) for the vehicles
        arrival_times = np.random.lognormal(mu, sigma, num_vehicles)
        arrival_times = arrival_times % 24  # Wrap around to fit within 24 hours

        # 3. Distribute Charging Demand Over Time

        # Create a time array
        time = np.arange(0, 24, self.time_step)  # time array with specified intervals
        power_demand = np.zeros_like(time)
        num_cars_plugged_in = np.zeros_like(time)

        for arrival_time, demand in zip(arrival_times, all_charging_demands):
            # Separate the values and the probabilities
            charging_powers = [item[0] for item in self.scenario_definition['Destination']['Charging power']]
            probabilities = [item[1] for item in self.scenario_definition['Destination']['Charging power']]

            # Randomly pick a value based on the specified probabilities
            charging_power = random.choices(charging_powers, weights=probabilities, k=1)[0]

            charging_duration = demand / charging_power
            start_idx = np.searchsorted(time, arrival_time)
            end_time = (arrival_time + charging_duration) % 24  # Wrap around to fit within 24 hours
            end_idx = np.searchsorted(time, end_time)

            if end_idx > start_idx:
                power_demand[start_idx:end_idx] += charging_power
                num_cars_plugged_in[start_idx:end_idx] += 1
            else:
                power_demand[start_idx:] += charging_power
                num_cars_plugged_in[start_idx:] += 1
                power_demand[:end_idx] += charging_power
                num_cars_plugged_in[:end_idx] += 1

            if charging_duration <= self.time_step:
                print("ALERT \t Charging duration is smaller than the timestep, this may lead to inaccurate results")

        # Convert power demand to MWh
        power_demand_mwh = power_demand / 1000  # converting kW to MW

        # Find the maximum number of cars plugged in at the same time
        max_cars_plugged_in = np.max(num_cars_plugged_in)

        self.charging_profile_destination = time, power_demand_mwh, num_cars_plugged_in, max_cars_plugged_in



