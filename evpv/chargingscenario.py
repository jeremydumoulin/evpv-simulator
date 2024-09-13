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

    def __init__(self, mobsim, ev_fleet, charging_efficiency, time_step, scenario_definition):
        self.mobsim = mobsim        

        self.ev_fleet = ev_fleet
        self.charging_efficiency = charging_efficiency 
        self.taz_properties = mobsim # Combine TAZ properties of each mobsim into a single effective TAZ properties df       

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

        # Correct the values based on the different vehicle share and occupancy 
        tmp_n_outflows = 0
        tmp_n_inflows = 0
        tmp_fkt_outflows = 0
        tmp_fkt_inflows = 0

        for vehicle in self.ev_fleet:
            tmp_n_outflows += result_df['n_outflows'] * vehicle[1] / vehicle[0]['vehicle_occupancy']
            tmp_n_inflows += result_df['n_inflows'] * vehicle[1] / vehicle[0]['vehicle_occupancy']
            tmp_fkt_outflows += result_df['fkt_outflows'] * vehicle[1] / vehicle[0]['vehicle_occupancy']
            tmp_fkt_inflows += result_df['fkt_inflows'] * vehicle[1] / vehicle[0]['vehicle_occupancy']

        result_df['n_outflows'] = tmp_n_outflows.apply(round)
        result_df['n_inflows'] = tmp_n_inflows.apply(round)
        result_df['fkt_outflows'] = tmp_fkt_outflows.apply(round)        
        result_df['fkt_inflows'] = tmp_fkt_inflows.apply(round)

        self._taz_properties = result_df

    # EV Fleet dictionnary
    @property
    def ev_fleet(self):
        return self._ev_fleet

    @ev_fleet.setter
    def ev_fleet(self, ev_fleet_value):
        self._ev_fleet = ev_fleet_value

    # Charging Efficiency
    @property
    def charging_efficiency(self):
        return self._charging_efficiency

    @charging_efficiency.setter
    def charging_efficiency(self, charging_efficiency_value):

        if charging_efficiency_value < 0.0 or charging_efficiency_value > 1.0:
            raise ValueError("The EV charging efficiency should be between 0 and 1")

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
    def scenario_definition(self, cs):
        if cs['Origin']['Share'] > 1.0 or cs['Origin']['Share'] < 0 or cs['Destination']['Share'] > 1.0 or cs['Destination']['Share'] < 0:
            raise ValueError("Share of charging at origin or destination should be between 0 and 1")

        if (cs['Origin']['Share'] + cs['Destination']['Share']) != 1.0:
            raise ValueError("The total of the shares at the origin and destination does sum up to")

        if cs['Origin']['Smart charging'] > 1.0 or cs['Origin']['Smart charging'] < 0 or cs['Destination']['Smart charging'] > 1.0 or cs['Destination']['Smart charging'] < 0 :
            raise ValueError("Share of smart charging should be between 0 and 1")

        if cs['Origin']['Arrival time'][0] > 24.0 or cs['Destination']['Arrival time'][0] > 24.0 or cs['Origin']['Arrival time'][0] < 0 or cs['Destination']['Arrival time'][0] < 0 :
            raise ValueError("Average arrival time should be between 0 and 24")

        self._scenario_definition = cs

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

            # Compute the vehicle share - weighted ev consumption 
            average_ev_consumption = 0
            for vehicle in self.ev_fleet:
                average_ev_consumption += vehicle[0]['ev_consumption'] * vehicle[1]

            Etot_origin = 2 * row['fkt_outflows'] * share_origin * average_ev_consumption / self.charging_efficiency # Multiply by 2 (origin-destination-origin)
            Etot_destination = 2 * row['fkt_inflows'] * share_destination * average_ev_consumption / self.charging_efficiency # Multiply by 2 (origin-destination-origin)

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

        travel_time_origin_destination_hours = self.scenario_definition['Travel time origin-destination']

        time_origin, power_profile_origin, num_cars_plugged_in_origin = self.eval_charging_profile(origin_or_destination = "Origin", travel_time_origin_destination_hours = travel_time_origin_destination_hours)
        time_destination, power_profile_destination, num_cars_plugged_in_destination = self.eval_charging_profile(origin_or_destination = "Destination", travel_time_origin_destination_hours = travel_time_origin_destination_hours)
        
        print(f" \t Max. number of vehicles charging simultaneously. At origin: {np.max(num_cars_plugged_in_origin)} - At destination: {np.max(num_cars_plugged_in_destination)}")
        print(f" \t Peak power. At origin: {np.max(power_profile_origin)} MW - At destination: {np.max(power_profile_destination)} MW")

        # Create DataFrames for each time series
        df = pd.DataFrame({
            'Time': time_origin,
            'Charging profile at origin (MW)': power_profile_origin,
            'Number of cars charging at origin': num_cars_plugged_in_origin,
            'Charging profile at destination (MW)': power_profile_destination,
            'Number of cars charging at destination': num_cars_plugged_in_destination,
            'Total charging profile (MW)': power_profile_origin + power_profile_destination
        })

        self._charging_profile = df

    @property
    def charging_profile(self):
        return self._charging_profile

    def eval_charging_profile(self, origin_or_destination = "Origin", travel_time_origin_destination_hours = 0.5):
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
        else:
            raise ValueError("Charging should be at origin or destination")

        """
        Initialize output variables
        """
        # Final time series
        time = np.arange(0, 24, self.time_step)  # time array with specified intervals
        power_demand = np.zeros_like(time)
        num_cars_plugged_in = np.zeros_like(time)

        # Charging demand and max charging power for all vehicles
        charging_demands = np.zeros(vehicle_counts)
        charging_powers = np.zeros(vehicle_counts)

        """
        Assign charging demand and charging power to each vehicle  
        """
        # Aggregate mobility flows 
        df_sum = self.mobsim[0].flows.copy()
        for mobsim in self.mobsim[1:]:
            df_sum = df_sum.add(mobsim.flows, fill_value=0)

        # Group by "Travel Distance (km)" and sum the "Flow" for each distance
        grouped_df = df_sum.groupby('Travel Distance (km)').agg({'Flow': 'sum'}).reset_index()

        # Calculate the total flow
        total_flow = grouped_df['Flow'].sum()

        # Calculate the probability for each travel distance
        grouped_df['Probability'] = grouped_df['Flow'] / total_flow

        # Extract distances and probabilities
        distances = grouped_df['Travel Distance (km)'].values
        distance_probabilities = grouped_df['Probability'].values

        # Calculate a charging demand and power for each type of vehicle

        tmp_charging_demands = [np.zeros(vehicle_counts)] * len(self.ev_fleet)
        tmp_charging_powers = [np.zeros(vehicle_counts)] * len(self.ev_fleet)

        # Loop over all vehicles a pick a random distribution of demand and charging power

        i = 0
        for vehicle in self.ev_fleet:
            tmp_charging_demands[i] = 2 * np.random.choice(distances, size=len(charging_demands), p=distance_probabilities) * vehicle[0]['ev_consumption'] / self.charging_efficiency

            # Randomly pick a charging power (depends on charging at origin or destination)
            if origin_or_destination == "Origin": 
                available_charging_power = [item[0] for item in vehicle[0]['charger_power']['Origin']]
                probabilities = [item[1] for item in vehicle[0]['charger_power']['Origin']]
            elif origin_or_destination == "Destination": 
                available_charging_power = [item[0] for item in vehicle[0]['charger_power']['Destination']]
                probabilities = [item[1] for item in vehicle[0]['charger_power']['Destination']]    

            tmp_charging_powers[i] = np.random.choice(available_charging_power, size=len(charging_demands), p=probabilities)

            i = i+1

        # Create a single list of charging demand and power based on the share of each vehicle

        # Probability of choosing a vechile
        probabilities = np.array([item[1] for item in self.ev_fleet])

        # Generate random choices based on probabilities
        # Create an array of shape (n, 3) to hold the lists
        choices = np.zeros((len(charging_demands), len(self.ev_fleet)), dtype=int)
        
        # Randomly choose one list per position to set as 1
        random_indices = np.random.choice(len(self.ev_fleet), size=len(charging_demands), p=probabilities)
        
        # Set the chosen list index to 1 for each position
        choices[np.arange((len(charging_demands))), random_indices] = 1

        num_lists = choices.shape[1]
        list_of_choices = [choices[:, i] for i in range(num_lists)]
        
        for i in range(len(tmp_charging_demands)):
            charging_demands += list_of_choices[i] * tmp_charging_demands[i]
            charging_powers += list_of_choices[i] * tmp_charging_powers[i]

        """
        Assign arrival time to each vehicle using lognormal distribution
        """
        # Lognormal parameters from arrival time average and standard deviation
        # Calculate mu and sigma for the lognormal distribution
        mean_arrival = arrival_time[0]
        stddev_arrival = arrival_time[1]

        mean_departure = departure_time[0]
        stddev_departure = departure_time[1]

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
            num_cars_plugged_in[start_indices[i]:end_indices[i]] += 1

        # Apply power demand for wrap-around cases
        for i in np.where(mask_wrap_around)[0]:
            if vehicles_with_smartcharging[i]:
                continue
            power_demand[start_indices[i]:] += charging_powers[i]
            num_cars_plugged_in[start_indices[i]:] += 1
            power_demand[:end_indices[i]] += charging_powers[i]
            num_cars_plugged_in[:end_indices[i]] += 1

        # Convert power demand to MWh
        power_demand_mw = power_demand / 1000  # converting kW to MW

        """
        Smart charging 
        """
        peak_power = 0  # Initialise le pic total de puissance à zéro
        power_demand = np.zeros(len(time))
        num_cars_plugged_in_smart = np.zeros(len(time))
        
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

            # Temporary array to track cars plugged in during uniform distribution
            temp_num_cars_plugged_in = np.zeros(len(time))
            
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
                    num_cars_plugged_in_smart[current_time_idx] += 1  # Increment number of cars charging
                    
                    # Si toute l'énergie est chargée, passer au véhicule suivant
                    if remaining_demand <= 0:
                        break
            
            # Si de l'énergie reste à charger, répartir uniformément
            if remaining_demand > 0:
                charge_power_uniform = remaining_demand / (departure_times[i] - arrival_times[i])
                for t in range(start_idx, end_idx):
                    current_time_idx = t % len(time)
                    power_demand[current_time_idx] += charge_power_uniform
                    temp_num_cars_plugged_in[current_time_idx] += 1  # Increment temp count for cars plugged in
        
            # Mettre à jour le pic total de puissance si nécessaire
            peak_power = np.max(power_demand)

            # Combine temp_num_cars_plugged_in with num_cars_plugged_in
            num_cars_plugged_in_smart = np.maximum(num_cars_plugged_in_smart, temp_num_cars_plugged_in)

        power_demand_mw = power_demand_mw + (power_demand / 1000)  # Convertir de kW en MW
        num_cars_plugged_in = num_cars_plugged_in + num_cars_plugged_in_smart

        return time, power_demand_mw, num_cars_plugged_in

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
