# coding: utf-8

""" 
ChargingDemand

A class to simulate the daily charging demand using mobility simulation and assumptions 
regarding the charging scenario and the EV fuel consumption and charging efficiency
"""

import numpy as np
import pandas as pd
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

class ChargingDemand:

    #######################################
    ############# ATTRIBUTES ##############
    #######################################
    
    # Mobility Simulation
    mobsim = None

    # EV fuel consumption and charging efficiency
    ev_consumption = .0
    charging_efficiency = .0

    # Charging scenario definition
    charging_scenario = {}

    # Outputs
    taz_properties = pd.DataFrame() # Relevant properties from Mobility Simulation for each TAZ

    charging_curve = pd.DataFrame() # Charging curve for EVs 
    charging_distribution = pd.DataFrame() # Charging need distribution across TAZs

    #######################################
    ############### METHODS ###############
    #######################################
    
    ############# Constructor #############
    ####################################### 

    def __init__(self, mobsim, ev_consumption, charging_efficiency):
        print("---")
        print(f"INFO \t ChargingDemand object initialisation")

        self.set_mobsim(mobsim)
        self.set_charging_efficiency(charging_efficiency)
        self.set_ev_consumption(ev_consumption)

        self.set_taz_properties()

        print(f"INFO \t ChargingDemand object created")
        print(f" \t - Number of trips: {self.taz_properties['n_inflows'].sum()} (n_in) // {self.taz_properties['n_outflows'].sum()} (n_out)")
        print(f" \t - FKT (origin to destination): {self.taz_properties['fkt_inflows'].sum()} km")
        print(f" \t - Average VKT (origin to destination): {self.taz_properties['fkt_inflows'].sum() / self.taz_properties['n_inflows'].sum()} km")

        print("---")

    ############# Setters #############
    ###################################

    def set_mobsim(self, mobsim):
        """ Setter for the mobsim attribute.
        """
        if not isinstance(mobsim, MobilitySim):
            print(f"ERROR \t A MobilitySim object is required as an input.")
            return

        # Check if trip generation has been performed
        if not 'n_outflows' in mobsim.traffic_zones.columns:
            print(f"ERROR \t MobilitySim object - Trip Generation has not been performed.")
            return

        # Check if trip generation has been performed
        if not 'Origin' in mobsim.flows.columns:
            print(f"ERROR \t MobilitySim object - Trip Distribution has not been performed.")
            return

        self.mobsim = mobsim

    def set_ev_consumption(self, ev_consumption):
        self.ev_consumption = ev_consumption

    def set_charging_efficiency(self, charging_efficiency):
        self.charging_efficiency = charging_efficiency

    def set_taz_properties(self):
        """ Setter for the taz_properties attribute.
        """
        taz = self.mobsim.traffic_zones
        flows = self.mobsim.flows

        # Prepare lists to hold the data
        taz_id = []
        geometric_center = []
        nearest_node = []
        bbox = []
        n_outflows = []
        n_inflows = []
        fkt_outflows = []
        fkt_inflows = []
        vkt_inflows = []
        vkt_outflows = []

        # Iterate over the TAZ and append data
        for index, row in taz.iterrows():
            taz_id.append(row['id'])
            geometric_center.append(row['geometric_center'])
            nearest_node.append(row['nearest_node'])
            bbox.append(row['bbox'])

            # Append values related to the origin (outflows)            
            out_df = flows[flows['Origin'] == row['id']].copy()
            out_df['Distance_Flow_Product'] = out_df['Travel Distance (km)'] * out_df['Flow']

            outflow_sum = out_df['Flow'].sum()
            distance_flow_product_sum_out = out_df['Distance_Flow_Product'].sum()

            n_outflows.append(outflow_sum)
            fkt_outflows.append(distance_flow_product_sum_out)
            if outflow_sum != 0:
                vkt_outflows.append(distance_flow_product_sum_out / outflow_sum)
            else:
                vkt_outflows.append(0)

            # Append values related to the destination (inflows)            
            in_df = flows[flows['Destination'] == row['id']].copy()
            in_df['Distance_Flow_Product'] = in_df['Travel Distance (km)'] * in_df['Flow']

            inflow_sum = in_df['Flow'].sum()
            distance_flow_product_sum_in = in_df['Distance_Flow_Product'].sum()

            n_inflows.append(inflow_sum)
            fkt_inflows.append(distance_flow_product_sum_in)
            if inflow_sum != 0:
                vkt_inflows.append(distance_flow_product_sum_in / inflow_sum)
            else:
                vkt_inflows.append(0)

            
        # Create the resulting DataFrame
        data = {
            'id': taz_id,
            'geometric_center': geometric_center,
            'nearest_node': nearest_node,
            'n_outflows': n_outflows,
            'n_inflows': n_inflows,
            'fkt_outflows': fkt_outflows,
            'fkt_inflows': fkt_inflows,
            'vkt_outflows': vkt_outflows,
            'vkt_inflows': vkt_inflows
        }

        taz_properties= pd.DataFrame(data)    

        self.taz_properties = taz_properties

    ######## Charging demand ##########
    ###################################

    def load_profile(self, mean_arrival_time, std_arrival_time, charging_power, time_step=0.1):
        """ 1. Load and Understand the Data
            2. Model Arrival Times with Lognormal Distribution
            3. Distribute Charging Demand Over Time
        """

        # 0. Parameters for lognormal distribution (mean and sigma)

        # Calculate sigma
        sigma = np.sqrt(np.log(1 + (std_arrival_time / mean_arrival_time)**2))
        # Calculate mu
        mu = np.log(mean_arrival_time) - 0.5 * sigma**2

        # 1. Load and Understand the Data

        # Expand the dataframe to list each vehicle and its charging demand
        vehicle_counts = self.taz_properties['n_outflows'].round().astype(int)
        charging_demands = 2 * self.taz_properties['vkt_outflows'] * self.ev_consumption / self.charging_efficiency  # Multiply by 2 (home-work-home)

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
        time = np.arange(0, 24, time_step)  # time array with specified intervals
        power_demand = np.zeros_like(time)
        num_cars_plugged_in = np.zeros_like(time)

        for arrival_time, demand in zip(arrival_times, all_charging_demands):
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

            if charging_duration <= time_step:
                print("ALERT \t Charging duration is smaller than the timestep, this may lead to inaccurate results")

        # Convert power demand to MWh
        power_demand_mwh = power_demand / 1000  # converting kW to MW

        # Find the maximum number of cars plugged in at the same time
        max_cars_plugged_in = np.max(num_cars_plugged_in)

        return time, power_demand_mwh, max_cars_plugged_in


