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
import math
import matplotlib.pyplot as plt
import folium
import branca.colormap as cm

from evpv import helpers as hlp
from evpv.mobilitysim import MobilitySim

class ChargingScenario:
    #######################################
    ############# Constructor #############
    #######################################

    def __init__(self, mobsim, ev_consumption, charging_efficiency, time_step, scenario_definition):
        self.mobsim = mobsim
        self.taz_properties = mobsim # Combine TAZ properties of each mobsim into a single effective TAZ properties df

        self.charging_efficiency =charging_efficiency
        self.ev_consumption = ev_consumption

        self.time_step = time_step
        self.scenario_definition = scenario_definition

        # Modeling results

        self._charging_demand = pd.DataFrame()
        self._charging_profile = pd.DataFrame()    

        # Printing results

        print(f"INFO \t ChargingDemand object created.")
        print(f" \t Number of trips: {self.taz_properties['n_inflows'].sum()} (n_in) // {self.taz_properties['n_outflows'].sum()} (n_out)")
        print(f" \t FKT (origin to destination): {self.taz_properties['fkt_inflows'].sum()} km")
        print(f" \t Average VKT (origin to destination): {self.taz_properties['fkt_inflows'].sum() / self.taz_properties['n_inflows'].sum()} km")  

        # Printing results
        self.spatial_charging_demand()
        self.temporal_charging_demand()      
        

    #######################################
    ###### Main Setters and Getters #######
    #######################################

    # Mobism
    @property
    def mobsim(self):
        return self._mobsim

    @mobsim.setter
    def mobsim(self, mobsims):
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
    def taz_properties(self):
        return self._taz_properties

    @taz_properties.setter
    def taz_properties(self, mobsim):
        # Create a list of TAZ properties
        df_list = []
        for mobsim_n in mobsim:
            df_list.append(mobsim_n.traffic_zones)

        # List of columns to sum
        columns_to_sum = ['n_outflows', 'n_inflows', 'fkt_outflows', 'fkt_inflows', 'vkt_outflows', 'vkt_inflows']

        # Initialize a DataFrame by summing the columns across all DataFrames in the list
        summed_df = pd.concat([df[columns_to_sum] for df in df_list]).groupby(level=0).sum()

        # Add back the non-summed columns from the first DataFrame (e.g., 'id')
        result_df = df_list[0][['id', 'geometric_center', 'bbox', 'is_within_target_area']].copy()
        result_df[columns_to_sum] = summed_df

        self._taz_properties = result_df

    # EV Consumption
    @property
    def ev_consumption(self):
        return self._ev_consumption

    @ev_consumption.setter
    def ev_consumption(self, ev_consumption_value):
        self._ev_consumption = ev_consumption_value

    # Charging Efficiency
    @property
    def charging_efficiency(self):
        return self._charging_efficiency

    @charging_efficiency.setter
    def charging_efficiency(self, charging_efficiency_value):
        self._charging_efficiency = charging_efficiency_value

    # Time step
    @property
    def time_step(self):
        return self._time_step

    @time_step.setter
    def time_step(self, time_step_value):
        self._time_step = time_step_value

    # Scenario definition
    @property
    def scenario_definition(self):
        return self._scenario_definition

    @scenario_definition.setter
    def scenario_definition(self, scenario_definition_value):
        self._scenario_definition= scenario_definition_value

    #######################################
    ####### Spatial Charging Demand #######
    #######################################

    def spatial_charging_demand(self):
        print(f"INFO \t Computing the spatial charging demand for each TAZ")

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
        
        self._charging_demand = df

        print(f" \t Charging demand. At origin: {self.charging_demand['Etot_origin_kWh'].sum()} kWh - At destination: {self.charging_demand['Etot_destination_kWh'].sum()} kWh")
        print(f" \t Vehicles charging. At origin: {self.charging_demand['n_vehicles_origin'].sum()} - At destination: {self.charging_demand['n_vehicles_destination'].sum()}")

    @property
    def charging_demand(self):
        return self._charging_demand

    #######################################
    ###### Temporal Charging Demand #######
    #######################################

    def temporal_charging_demand(self):
        print(f"INFO \t Evaluating temporal charging profile")

        time_origin, power_profile_origin, num_cars_plugged_in_origin, max_cars_plugged_in_origin = self.eval_charging_profile_origin()
        time_destination, power_profile_destination, num_cars_plugged_in_destination, max_cars_plugged_in_destination = self.eval_charging_profile_destination()
        
        print(f" \t Max. number of vehicles charging simultaneously. At origin: {max_cars_plugged_in_origin} - At destination: {max_cars_plugged_in_destination}")
        print(f" \t Peak power. At origin: {np.max(power_profile_origin)} MW - At destination: {np.max(power_profile_destination)} MW")

        # Create DataFrames for each time series
        df = pd.DataFrame({
            'Time': time_origin,
            'Power Profile Origin (MW)': power_profile_origin,
            'Num Cars Plugged In Origin': num_cars_plugged_in_origin,
            'Max Cars Plugged In Origin': max_cars_plugged_in_origin,
            'Power Profile Destination (MW)': power_profile_destination,
            'Num Cars Plugged In Destination': num_cars_plugged_in_destination,
            'Total profile (MW)': power_profile_origin + power_profile_destination
        })

        self._charging_profile = df

    @property
    def charging_profile(self):
        return self._charging_profile

    def eval_charging_profile_origin(self):
        """ 1. Load and Understand the Data
            2. Model Arrival Times with Lognormal Distribution
            3. Distribute Charging Demand Over Time
        """
        print(f"INFO \t ... Charging at origin...")

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
        
        return time, power_demand_mwh, num_cars_plugged_in, max_cars_plugged_in

    def eval_charging_profile_destination(self):
        """ 1. Load and Understand the Data
            2. Model Arrival Times with Lognormal Distribution
            3. Distribute Charging Demand Over Time
        """
        print(f"INFO \t ... Charging at destination...")

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

        return time, power_demand_mwh, num_cars_plugged_in, max_cars_plugged_in

    #######################################
    ### Post-processing & visualisation ###
    #######################################

    def chargingdemand_total_to_map(self):
        df = self.charging_demand

        # 1. Create an empty map

        m = folium.Map(location=self.mobsim[0].centroid_coords, zoom_start=12, tiles='CartoDB Positron', control_scale=True) # Create the map

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

        # Add Layer Control and Save 

        folium.LayerControl().add_to(m)

        return m

    def chargingdemand_pervehicle_to_map(self):
        df = self.charging_demand

        # 1. Create an empty map

        m = folium.Map(location=self.mobsim[0].centroid_coords, zoom_start=12, tiles='CartoDB Positron', control_scale=True) # Create the map

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
        linear = cm.LinearColormap(["#edf8b1", "#7fcdbb", "#2c7fb8"], vmin=0, vmax=max(df['E0_origin_kWh'].max(),df['E0_destination_kWh'].max()) )

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

        # Add the color scale legend to the map
        linear.caption = 'Charging needs (kWh/car)'
        linear.add_to(m)

        # Add Layer Control and Save 

        folium.LayerControl().add_to(m)

        return m

    def chargingdemand_nvehicles_to_map(self):
        df = self.charging_demand

        # 1. Create an empty map

        m = folium.Map(location=self.mobsim[0].centroid_coords, zoom_start=12, tiles='CartoDB Positron', control_scale=True) # Create the map

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
        linear = cm.LinearColormap(["#edf8b1", "#7fcdbb", "#2c7fb8"], vmin=0, vmax=max(df['n_vehicles_origin'].max(), df['n_vehicles_destination'].max()) )

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

        # Add the color scale legend to the map
        linear.caption = 'Number of vehicles'
        linear.add_to(m)

        # Add Layer Control and Save 

        folium.LayerControl().add_to(m)

        return m
