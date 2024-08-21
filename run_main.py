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

percentage_population_to_ignore = 0

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

use_cached_data = False

#############################################
### MOBILITY SIMULATION (home-work-home) ####
#############################################

# MobilitySim 

if os.path.isfile(OUTPUT_PATH / "evpv_Tmp_MobilitySim_Cache.pkl"):
    mobsim = MobilitySim.from_pickle(OUTPUT_PATH / "evpv_Tmp_MobilitySim_Cache.pkl")
        
else:
    mobsim = MobilitySim(
        target_area = shapefile_path,
        population_density = population_density_path, 
        destinations = destinations_path,
        pickle_file = OUTPUT_PATH / "evpv_Tmp_MobilitySim_Cache.pkl")

    mobsim.setup_simulation(taz_target_width_km = 5, simulation_area_extension_km = 0, population_to_ignore_share = 0.05)
    mobsim.trip_generation(n_trips_per_inhabitant = n_trips_per_inhabitant)     
    mobsim.trip_distribution(model = model, attraction_feature = attraction_feature, cost_feature = cost_feature, vkt_offset = vkt_offset)

    mobsim.to_pickle(OUTPUT_PATH / f"evpv_Tmp_MobilitySim_Cache.pkl")

# 4. Storing outputs

# All flows and TAZ properties

mobsim.flows.to_csv(OUTPUT_PATH / "evpv_Result_MobilitySim_OriginDestinationFlows.csv", index=False) # Store aggregated TAZ features as csv
mobsim.traffic_zones.to_csv(OUTPUT_PATH / "evpv_Result_MobilitySim_TrafficAnalysisZones.csv", index=False) # Store aggregated TAZ features as csv

# Histogram of VKTs 

centroid_distance = mobsim.flows['Centroid Distance (km)']
travel_distance = mobsim.flows['Travel Distance (km)']
weights = mobsim.flows['Flow']

# Calculate the histogram
centroid_distance_counts, bin_edges = np.histogram(centroid_distance, bins=200, weights=weights)
travel_distance_counts, bin_edges = np.histogram(travel_distance, bins=200, weights=weights)

# Calculate the bin centers (optional)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Create a DataFrame to store the bin centers and counts
hist_df = pd.DataFrame({
    'Distance (km)': bin_centers,
    'Centroid ': centroid_distance_counts,
    'Travel (road)': travel_distance_counts
})

# Save to CSV
hist_df.to_csv(OUTPUT_PATH / "evpv_Result_MobilitySim_DistanceDistribution.csv", index=False)

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

# 1. Create an empty map
m1 = folium.Map(location=mobsim.centroid_coords, zoom_start=12, tiles='CartoDB Positron', control_scale=True) # Create the map

# 2. Add administrative boundaries

# Define style function to only show lines
def style_function(feature):
    return {
        'color': 'blue',  # Set line color
        'weight': 3,      # Set line weight
        'fillColor': 'none',  # Set fill color to 'none'
    }

folium.GeoJson(mobsim.target_area_shapefile['features'][0]['geometry'], name='Administrative boundary', style_function=style_function).add_to(m1)

# 3. Add Simulation bbox

minx, miny, maxx, maxy = mobsim.simulation_bbox

# Create a rectangle using the bounding box coordinates
rectangle = folium.Rectangle(
    bounds=[[miny, minx], [maxy, maxx]],
    fill=True,  # Fill the rectangle
    fill_opacity=0,  # Set the opacity of the fill color
    color='blue',  # Border color
    weight=2,  # Border width
)

# Create a feature group to hold the rectangle and give it a name
feature_group = folium.FeatureGroup(name='Simulation Area')
rectangle.add_to(feature_group)
feature_group.add_to(m1)

# 4. Add Population data

m1 = hlp.add_raster_to_folium(mobsim.population_density, m1)

# 5. Add TAZs

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
    ).add_to(m1)

# # Apply the function to each row in the DataFrame
mobsim.traffic_zones.apply(add_rectangle, axis=1)

# Get TAZ data
df = mobsim.traffic_zones

# Add center points

# Add markers

for idx, row in df.iterrows():
    lat, lon = row['geometric_center']
    folium.Marker(
        location=[lon, lat],
        icon=folium.Icon(color='red'),
        popup=f"ID: {row['id']} - ({lat}, {lon}) - Pop: {int(row['population'])} - Dest: {int(row['destinations'])}"
    ).add_to(m1)

# Add destinations

# Normalize data for color scaling
linear = cm.LinearColormap(["white", "yellow", "red"], vmin=df['destinations'].min(), vmax=df['destinations'].max())

# Create a feature group for all polygons
feature_group = folium.FeatureGroup(name='Destinations')

# Add polygons to the feature group
for idx, row in df.iterrows():
    bbox_polygon = row['bbox']
    bbox_coords = bbox_polygon.bounds

    # Create a rectangle for each row
    rectangle = folium.Rectangle(
        bounds=[(bbox_coords[1], bbox_coords[0]), (bbox_coords[3], bbox_coords[2])],
        color=None,
        fill=True,
        fill_color=linear(row['destinations']),
        fill_opacity=0.7,
    )

    # Add the rectangle to the feature group
    rectangle.add_to(feature_group)

# Add the feature group to the map
feature_group.add_to(m1)

# 6. Add Aggregateds Population

# Normalize population data for color scaling
linear = cm.LinearColormap(["white", "yellow", "red"], vmin=df['population'].min(), vmax=df['population'].max())

# Create a feature group for all polygons
feature_group = folium.FeatureGroup(name='Population')

# Add polygons to the feature group
for idx, row in df.iterrows():
    bbox_polygon = row['bbox']
    bbox_coords = bbox_polygon.bounds

    # Create a rectangle for each row
    rectangle = folium.Rectangle(
        bounds=[(bbox_coords[1], bbox_coords[0]), (bbox_coords[3], bbox_coords[2])],
        color=None,
        fill=True,
        fill_color=linear(row['population']),
        fill_opacity=0.7,
    )

    # Add the rectangle to the feature group
    rectangle.add_to(feature_group)

# Add the feature group to the map
feature_group.add_to(m1)

# Add Layer Control and Save 

folium.LayerControl().add_to(m1)
m1.save(OUTPUT_PATH / "evpv_Result_MobilitySim_MainGeoInputs.html")

###### Folium map with trip generation ######
#############################################

# 1. Create an empty map
m2 = folium.Map(location=mobsim.centroid_coords, zoom_start=12, tiles='CartoDB Positron', control_scale=True) # Create the map

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
    ).add_to(m2)

# Apply the function to each row in the DataFrame
mobsim.traffic_zones.apply(add_rectangle, axis=1)

# 3. Add number of outflows

# Normalize data for color scaling
linear = cm.LinearColormap(["white", "yellow", "red"], vmin=df['n_outflows'].min(), vmax=df['n_outflows'].max())

# Create a feature group for all polygons
feature_group = folium.FeatureGroup(name='Number of trips')

# Add polygons to the feature group
for idx, row in df.iterrows():
    bbox_polygon = row['bbox']
    bbox_coords = bbox_polygon.bounds

    # Create a rectangle for each row
    rectangle = folium.Rectangle(
        bounds=[(bbox_coords[1], bbox_coords[0]), (bbox_coords[3], bbox_coords[2])],
        color=None,
        fill=True,
        fill_color=linear(row['n_outflows']),
        fill_opacity=0.7,
        popup=f"ID: {row['id']} - Trips: {int(row['n_outflows'])}"
    )

    # Add the rectangle to the feature group
    rectangle.add_to(feature_group)

# Add the feature group to the map
feature_group.add_to(m2)

# Add the color scale legend to the map
linear.caption = 'Number of Trips (n_outflows)'
linear.add_to(m2)

# Add Layer Control and Save 

folium.LayerControl().add_to(m2)
m2.save(OUTPUT_PATH / "evpv_Result_MobilitySim_TripGeneration.html")

##### Folium map with trip distribution #####
#############################################

# 1. Create an empty map

m3 = folium.Map(location=mobsim.centroid_coords, zoom_start=12, tiles='CartoDB Positron', control_scale=True) # Create the map

# 2. Group the dataframe by Origin

grouped = mobsim.flows.groupby('Origin')

# 3. Iterate over each origin and create a FeatureGroup

for origin_id, group in grouped:
    feature_group = folium.FeatureGroup(name=f'Origin: {origin_id}')
    
    # Add flows to the feature group
    for idx, row in group.iterrows():
        linear = cm.LinearColormap(["white", "yellow", "red"], vmin=group['Flow'].min(), vmax=group['Flow'].max())

        bbox_polygon = mobsim.traffic_zones[mobsim.traffic_zones['id'] == row['Destination']]['bbox'].values[0]
        #print(bbox_polygon[1])
        bbox_coords = bbox_polygon.bounds

        folium.Rectangle(
            bounds=[(bbox_coords[1], bbox_coords[0]), (bbox_coords[3], bbox_coords[2])],
            color=None,
            fill=True,
            fill_color=linear(row.Flow),
            fill_opacity=0.7,
            tooltip=f'Commuters: {row.Flow} - Car trips: {row.Flow} '
        ).add_to(feature_group)
    
    # Add the feature group to the map
    feature_group.add_to(m3)

# Add Layer Control and Save

folium.LayerControl().add_to(m3)
m3.save(OUTPUT_PATH / "evpv_Result_MobilitySim_TripDistribution.html")

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


