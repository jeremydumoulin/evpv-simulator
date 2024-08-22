# coding: utf-8

""" 
A python script to simulate the spatio-temporal charging demand based on 
mobility simulation.
"""

import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import LineString, Point, box, Polygon
from shapely import wkt
from shapely.ops import transform
from rasterio.features import geometry_mask
from rasterio.features import shapes
from shapely.geometry import mapping
import pyproj
from pyproj import Transformer
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from branca.colormap import LinearColormap
import numpy as np
import math
import os
import folium
from folium.plugins import AntPath
from folium.plugins import HeatMap
import pickle

from dotenv import load_dotenv
from pathlib import Path
import osmnx as ox
import branca.colormap as cm

from evpv.mobilitysim import MobilitySim
from evpv.chargingscenario import ChargingScenario
from evpv import helpers as hlp

#############################################
# PARAMETERS - MODIFY ACCORDING TO YOUR NEEDS
#############################################

"""
Environment variables
"""

load_dotenv() # take environment variables from .env

INPUT_PATH = Path( str(os.getenv("INPUT_PATH")) )
OUTPUT_PATH = Path( str(os.getenv("OUTPUT_PATH")) )

"""
Global parameters 
"""

shapefile_path = INPUT_PATH / "gadm41_ETH_1_AddisAbeba.json" # Addis Ababa administrative boundaries
population_density_path = INPUT_PATH / "GHS_POP_merged_4326_3ss_V1_0_R8andR9_C22_cropped.tif" # Population density raster
destinations_path = INPUT_PATH /  "workplaces.csv"

taz_target_width_km = 3 # Desired TAZ width
simulation_area_extension_km = 0
population_to_ignore = 0

share_active = 0.1
share_unemployed = 0.227
share_home_office = 0.0
mode_share = 1.0
vehicle_occupancy = 1.2

n_trips_per_inhabitant = (share_active * (1 - share_unemployed) * (1 - share_home_office)) *  mode_share / vehicle_occupancy

vkt_offset = 0
model = "gravity_exp_016"
attraction_feature = "destinations"
cost_feature = "distance_road"

use_cached_data = True

#############################################
## MOBILITY SIMULATION 1 (home-work-home) ###
#############################################

# MobilitySim 

if os.path.isfile(OUTPUT_PATH / "evpv_Tmp_MobilitySim_Cache.pkl") and use_cached_data:
    mobsim = MobilitySim.from_pickle(OUTPUT_PATH / "evpv_Tmp_MobilitySim_Cache.pkl")

else:
    mobsim = MobilitySim(
        target_area = shapefile_path,
        population_density = population_density_path, 
        destinations = destinations_path)

    mobsim.setup_simulation(taz_target_width_km = taz_target_width_km, simulation_area_extension_km = simulation_area_extension_km, population_to_ignore_share = population_to_ignore)
    mobsim.trip_generation(n_trips_per_inhabitant = n_trips_per_inhabitant)     
    mobsim.trip_distribution(model = model, ors_key = os.getenv("ORS_KEY"), attraction_feature = attraction_feature, cost_feature = cost_feature, vkt_offset = vkt_offset)

    mobsim.to_pickle(OUTPUT_PATH / f"evpv_Tmp_MobilitySim_Cache.pkl")

# Printing FKT and VKT
print(f"FKT = ({mobsim.fkt} +/- {mobsim.fkt_error}) km")
print(f"VKT = ({mobsim.vkt} +/- {mobsim.vkt_error}) km")

# Storing outputs

# All flows and TAZ properties

mobsim.flows.to_csv(OUTPUT_PATH / "evpv_Result_MobilitySim_OriginDestinationFlows.csv", index=False) # Store aggregated TAZ features as csv
mobsim.traffic_zones.to_csv(OUTPUT_PATH / "evpv_Result_MobilitySim_TrafficAnalysisZones.csv", index=False) # Store aggregated TAZ features as csv

# Histogram of VKTs 

vkt_distribution = mobsim.vkt_histogram(n_bins = 200)
vkt_distribution.to_csv(OUTPUT_PATH / "evpv_Result_MobilitySim_VKThistogram.csv", index=False)

# Maps
#mobsim.setup_to_map().save(OUTPUT_PATH / "evpv_Result_MobilitySim_SimulationSetup.html")
mobsim.trip_generation_to_map().save(OUTPUT_PATH / "evpv_Result_MobilitySim_TripGeneration.html")
mobsim.trip_distribution_to_map(trip_id = "4_4").save(OUTPUT_PATH / "evpv_Result_MobilitySim_TripDistribution.html")

#############################################
############### CHARGING NEEDS ##############
#############################################

chargedem = ChargingScenario(
    mobsim = [mobsim],
    ev_consumption = 0.2,
    charging_efficiency = 0.9,
    time_step = 1/60,
    scenario_definition = {
    "Origin": {
        "Share": 0.8, # Charging location share
        "Charging power": [[3.7, 0.75], [11, 0.25]], # Charging powers and shares of each charger
        "Charging time": [18, 2] # Average charger plugin time and std deviation
    },
    "Destination": {
        "Share": 0.2,
        "Charging power": [[11, 1]], 
        "Charging time": [10, 2]
    }
})

# Store aggregated TAZ features as csv
chargedem.charging_demand.to_csv(OUTPUT_PATH / "evpv_Result_ChargingDemand.csv", index=False) 

# Store charging curve
time_origin, power_profile_origin, num_cars_plugged_in_origin, max_cars_plugged_in_origin = chargedem.charging_profile_origin
time_destination, power_profile_destination, num_cars_plugged_in_destination, max_cars_plugged_in_destination = chargedem.charging_profile_destination

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

# Save to CSV
df.to_csv(OUTPUT_PATH / "evpv_Result_ChargingDemand_PowerProfile.csv", index=False)

#############################################
################ VISUALISATION ##############
#############################################

###### Folium map with main geo inputs ######
#############################################


###### Folium map with trip generation ######
#############################################



##### Folium map with trip distribution #####
#############################################

########## Charging needs per TAZ ###########
#############################################

df = chargedem.charging_demand

# 1. Create an empty map

m4 = folium.Map(location=mobsim.centroid_coords, zoom_start=12, tiles='CartoDB Positron', control_scale=True) # Create the map

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
    ).add_to(m4)

# Apply the function to each row in the DataFrame
df.apply(add_rectangle, axis=1)

# 3. Charging AT ORIGIN

# Normalize data for color scaling
linear = cm.LinearColormap(["white", "yellow", "red"], vmin=df['Etot_origin_kWh'].min(), vmax=df['Etot_origin_kWh'].max())

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
feature_group.add_to(m4)

# 4. Charging AT DESTINATION

# Normalize data for color scaling
#linear = cm.LinearColormap(["white", "yellow", "red"], vmin=df['Etot_destination_kWh'].min(), vmax=df['Etot_destination_kWh'].max())

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
feature_group.add_to(m4)

# Add the color scale legend to the map
linear.caption = 'Charging needs (kWh)'
linear.add_to(m4)

# Add Layer Control and Save 

folium.LayerControl().add_to(m4)
m4.save(OUTPUT_PATH / "evpv_Result_ChargingDemand_Etot.html")

### Charging needs per vehicle per TAZ ######
#############################################

df = chargedem.charging_demand

# 1. Create an empty map

m5 = folium.Map(location=mobsim.centroid_coords, zoom_start=12, tiles='CartoDB Positron', control_scale=True) # Create the map

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
    ).add_to(m5)

# Apply the function to each row in the DataFrame
df.apply(add_rectangle, axis=1)

# 3. Charging AT ORIGIN

# Normalize data for color scaling
linear = cm.LinearColormap(["white", "yellow", "red"], vmin=df['E0_origin_kWh'].min(), vmax=df['E0_origin_kWh'].max())

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
feature_group.add_to(m5)

# 4. Charging AT DESTINATION

# Normalize data for color scaling
#linear = cm.LinearColormap(["white", "yellow", "red"], vmin=df['Etot_destination_kWh'].min(), vmax=df['Etot_destination_kWh'].max())

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
feature_group.add_to(m5)

# Add the color scale legend to the map
linear.caption = 'Charging needs (kWh/car)'
linear.add_to(m5)

# Add Layer Control and Save 

folium.LayerControl().add_to(m5)
m5.save(OUTPUT_PATH / "evpv_Result_ChargingDemand_E0.html")


