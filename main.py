# coding: utf-8

""" 
A python script to simulate the spatio-temporal charging demand based on 
mobility simulation.
"""

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point, box, Polygon
from shapely.ops import transform
import pyproj
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import math
import os
import folium
from folium.plugins import AntPath

from dotenv import load_dotenv
from pathlib import Path
import osmnx as ox
import branca.colormap as cm

from evpv.mobilitysim import MobilitySim
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

shapefile_path = INPUT_PATH / "gadm41_ETH_1.json" # Addis Ababa administrative boundaries
population_density_path = INPUT_PATH / "GHS_POP_merged_4326_3ss_V1_0_R8andR9_C22.tif" # Population density raster

buffer_distance = 0 # Margin in km added to the bbox around the shapefile_path
n_subdivisions = 15 # Number of subdivisions of the bbox to create traffic analysis zones
road_network_filter_string = '["highway"!~"^(service|track|residential)$"]' # Roads used in the road network
workplaces_tags = { # Tags used to get workplaces
            "building": ["industrial", "office"],
            "company": [],
            "landuse": ["industrial"],
            "industrial": [],
            "office": ["company", "government"],
            "amenity": ["university", "research_institute", "conference_centre", "bank", "hospital", "townhall", "police", "fire_station", "post_office", "post_depot"]
        }

#####################
# Mobility simulation
#####################

"""
##################### MobilitySim object initialisation #####################
"""

# 1. Creating the object 

mobsim = MobilitySim(
    target_area_shapefile = shapefile_path,
    population_density = population_density_path, 
    buffer_distance = buffer_distance, 
    n_subdivisions = n_subdivisions,
    road_network_filter_string = road_network_filter_string,
    workplaces_tags = workplaces_tags)

mobsim.traffic_zones.to_csv(OUTPUT_PATH / "traffic_zones.csv", index=False)

# 2. Pre-analysis: number of workplaces as a function of the population
# Is the population a good proxy for workplaces?

# plt.figure(figsize=(10, 6))
# plt.scatter(mobsim.traffic_zones['population'], mobsim.traffic_zones['workplaces'], color='blue')

# # Adding title and labels
# plt.title('Number of Workplaces as a Function of Population')
# plt.xlabel('Population')
# plt.ylabel('Number of Workplaces')

# plt.show()
# plt.savefig(OUTPUT_PATH / 'population_vs_workplaces.png')

# 3. Create a folium map and add all the data

mymap = folium.Map(location=mobsim.centroid_coords, zoom_start=12, tiles='CartoDB Positron', control_scale=True) # Create the map


##### 3.1 Administrative boundaries

# Define style function to only show lines
def style_function(feature):
    return {
        'color': 'blue',  # Set line color
        'weight': 3,      # Set line weight
        'fillColor': 'none',  # Set fill color to 'none'
    }

folium.GeoJson(mobsim.target_area_shapefile['features'][0]['geometry'], name='Administrative boundary', style_function=style_function).add_to(mymap)

##### 3.2 Simulation bbox

minx, miny, maxx, maxy = mobsim.simulation_bbox

# Create a rectangle using the bounding box coordinates
rectangle = folium.Rectangle(
    bounds=[[miny, minx], [maxy, maxx]],
    fill=True,  # Fill the rectangle
    fill_opacity=0,  # Set the opacity of the fill color
    color='blue',  # Border color
    weight=2,  # Border width
)

# Add the rectangle to the map
rectangle.add_to(mymap)

##### 3.3 Population data

mymap = hlp.add_raster_to_folium(mobsim.population_density, mymap)

##### 3.4 Road network

# # Convert the network to a GeoDataFrame
# gdf = ox.graph_to_gdfs(mobsim.road_network, nodes=False, edges=True)

# # Add the road network to the map /!\ Heavy process
# folium.GeoJson(gdf, name='Road Network', style_function=lambda x:{'fillColor': '#000000', 'color': '#000000'}).add_to(mymap)

##### 3.5 Workplaces

# Add workplaces
# Add markers for each center point
# for point in mobsim.workplaces:
#     folium.Marker(location=[point[1], point[0]], popup="Center point").add_to(mymap)

##### 3.6 Markers for the nearest nodes

for idx, row in mobsim.traffic_zones.iterrows():
    #nearest_node_lat, nearest_node_lon = row['nearest_node']
    nearest_node_lat, nearest_node_lon = row['geometric_center']
    folium.Marker(
        location=[nearest_node_lon, nearest_node_lat],
        icon=folium.Icon(color='red'),
        popup=f"{nearest_node_lat}, {nearest_node_lon} - Pop: {int(row['population'])} - Work: {int(row['workplaces'])}"
    ).add_to(mymap)

##### 3.7 TAZs

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
    ).add_to(mymap)

# # Apply the function to each row in the DataFrame
mobsim.traffic_zones.apply(add_rectangle, axis=1)

# # Display the map
folium.LayerControl().add_to(mymap)
mymap.save(OUTPUT_PATH / "data_map.html")


"""
##################### Trip generation #####################
"""

# Generation of number of home-work-home and home-study-home trips by car 

mobsim.trip_generation(
    share_active = 0.76, 
    share_unemployed = 0.227, 
    share_home_office = 0.0, 
    mode_share = 1.0, 
    vehicle_occupancy = 1.0
)

# Creating a map with the number of commuters

df = mobsim.traffic_zones

m = folium.Map(location=mobsim.centroid_coords, zoom_start=12, tiles='CartoDB Positron') # Create the map

# Normalize population data for color scaling
linear = cm.LinearColormap(["green", "yellow", "red"], vmin=df['n_commuters'].min(), vmax=df['n_commuters'].max())

# Add polygons to the map
for idx, row in df.iterrows():

    bbox_polygon = row['bbox']
    bbox_coords = bbox_polygon.bounds

    folium.Rectangle(
        bounds=[(bbox_coords[1], bbox_coords[0]), (bbox_coords[3], bbox_coords[2])],
        color=None,
        fill=True,
        fill_color=linear(row.n_commuters),
        fill_opacity=0.7,
        tooltip=f'Commuters: {row.n_commuters} - Car trips: {row.n_trips} '
    ).add_to(m)

# Display the map
m.save(OUTPUT_PATH / 'n_commuters.html')

"""
##################### Trip distribution #####################
"""

mobsim.trip_distribution(model = "radiation", attraction_feature = "workplaces", cost_feature = "distance_centroid", taz_center = "centroid")

df = mobsim.flows

df.to_csv(OUTPUT_PATH / "mobility_flows.csv", index=False)

flow_weighted_avg_distance = np.average(df['Travel Distance (km)'], weights=df['Flow'])
flow_weighted_avg_centroid = np.average(df['Centroid Distance (km)'], weights=df['Flow'])

print("Flow-weighted average travel distance:", flow_weighted_avg_distance)
print("Flow-weighted average travel distance btwn centroids:", flow_weighted_avg_centroid)

print("Total flow:", df['Flow'].sum())
print("Total trips:", mobsim.traffic_zones['n_trips'].sum())

# print("Flow-weighted average travel time:", flow_weighted_avg_time)

# df = mobsim.flows_car
# print(mobsim.flows_car)

# # Calculate flow-weighted average travel time and distance
# flow_weighted_avg_time = np.average(df['Travel Time (min)'], weights=df['Flow'])
# flow_weighted_avg_distance = np.average(df['Travel Distance (km)'], weights=df['Flow'])

# print("Flow-weighted average travel time:", flow_weighted_avg_time)
# print("Flow-weighted average travel distance:", flow_weighted_avg_distance)

#Plot histogram of total flow for each bin of travel time
plt.hist(df['Centroid Distance (km)'], bins=200, weights=df['Flow'], color='blue', edgecolor='black')
plt.xlabel('Centroid Distance (km)')
plt.ylabel('Total Flow')
plt.title('Total Flow as a function of Centroid Distance (km)')
plt.grid(True)
plt.show()


# df.to_csv('flow_distanc.csv', index=False) 

# Expand coordinates_df into separate latitude and longitude columns
# mobsim.traffic_zones[['longitude', 'latitude']] = pd.DataFrame(mobsim.traffic_zones['geometric_center'].tolist(), index=mobsim.traffic_zones.index) 

# # Merge flows_df with coordinates_df to get coordinates for origins and destinations
# merged_df = mobsim.flows_car.merge(mobsim.traffic_zones, left_on='Origin', right_on='id').rename(columns={'latitude': 'origin_lat', 'longitude': 'origin_lon'})
# merged_df = merged_df.merge(mobsim.traffic_zones, left_on='Destination', right_on='id').rename(columns={'latitude': 'dest_lat', 'longitude': 'dest_lon'})

# # Create a folium map centered around the average coordinates
# mymap = folium.Map(location=[merged_df['origin_lat'].mean(), merged_df['origin_lon'].mean()], zoom_start=13)

# # Apply the function to each row in the DataFrame
# mobsim.traffic_zones.apply(add_rectangle, axis=1)

# # Function to add lines to the map
# # Function to add lines to the map
# def add_flow_line(row):
#     if row['Flow'] > 1000:  # Only add flows with non-zero value
#         origin = (row['origin_lat'], row['origin_lon'])
#         destination = (row['dest_lat'], row['dest_lon'])
#         # Create polyline with arrows at the end
#         folium.plugins.AntPath(
#             locations=[origin, destination],
#             color='red',
#             use_arrows=True,  # Display arrows at the end
#             delay=1000000000,  # Delay between each arrow
#             dash_array=[10, 20],  # Dash pattern to make the arrow part dashed
#             weight=20,
#             opacity=0.6
#         ).add_to(mymap)


# # Apply the function to each row in the DataFrame
# merged_df.apply(add_flow_line, axis=1)



# Save the map to an HTML file
# mymap.save('flow_map.html')


