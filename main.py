# coding: utf-8

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
from dotenv import load_dotenv
from pathlib import Path
import osmnx as ox
import branca.colormap as cm

from evpv.mobilitysim import MobilitySim
from evpv import helpers as hlp

"""
Environment variables
"""
load_dotenv() # take environment variables from .env

INPUT_PATH = Path( str(os.getenv("INPUT_PATH")) )
OUTPUT_PATH = Path( str(os.getenv("OUTPUT_PATH")) )

"""
Parameters
"""

shapefile_path = INPUT_PATH / "gadm41_ETH_1.json"
population_density_path = INPUT_PATH / "GHS_POP_merged_4326_3ss_V1_0_R8andR9_C22.tif"

"""
Pre-processing
"""
# Crop the population raster to the bbox



"""
Mobility simulation
"""

"""
Initialisation
"""

# Initialize mobility simulation
mobsim = MobilitySim(shapefile_path, population_density = population_density_path, buffer_distance = 0, n_subdivisions = 15)

# print(mobsim.centroid_coords)
# print(mobsim.simulation_bbox)

#print(mobsim.mobility_zones)
#print(mobsim.mobility_zones['population'].sum())


# Plot the boundaries on a folium map

# Define style function to only show lines
def style_function(feature):
    return {
        'color': 'blue',  # Set line color
        'weight': 3,      # Set line weight
        'fillColor': 'none',  # Set fill color to 'none'
    }

mymap = folium.Map(location=mobsim.centroid_coords, zoom_start=12, tiles='CartoDB Positron') # Create the map


# Add GeoJSON layer to the map, specifying style function
folium.GeoJson(mobsim.target_area_shapefile['features'][0]['geometry'], name='Administrative boundary', style_function=style_function).add_to(mymap)

minx, miny, maxx, maxy = mobsim.simulation_bbox

# Create a rectangle using the bounding box coordinates
rectangle = folium.Rectangle(
    bounds=[[miny, minx], [maxy, maxx]],
    fill=True,  # Fill the rectangle
    fill_opacity=0.2,  # Set the opacity of the fill color
    color='blue',  # Border color
    weight=2,  # Border width
)

# Add the rectangle to the map
rectangle.add_to(mymap)

# Add population data to map

mymap = hlp.add_raster_to_folium(mobsim.population_density, mymap)


# Add road network

# Convert the network to a GeoDataFrame
gdf = ox.graph_to_gdfs(mobsim.road_network, nodes=False, edges=True)

# Add the road network to the map /!\ Heavy process
folium.GeoJson(gdf, name='Road Network', style_function=lambda x:{'fillColor': '#000000', 'color': '#000000'}).add_to(mymap)


# Add workplaces
# Add markers for each center point
# for point in mobsim.workplaces:
#     folium.Marker(location=[point[1], point[0]], popup="Center point").add_to(mymap)


# Add markers for the nearest nodes
for idx, row in mobsim.mobility_zones.iterrows():
    nearest_node_lat, nearest_node_lon = row['nearest_node']
    folium.Marker(
        location=[nearest_node_lon, nearest_node_lat],
        icon=folium.Icon(color='red'),
        popup=f"{nearest_node_lat}, {nearest_node_lon} - Pop: {int(row['population'])} - Work: {int(row['workplaces'])}"
    ).add_to(mymap)


# Add the subdivisions

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


# Apply the function to each row in the DataFrame
mobsim.mobility_zones.apply(add_rectangle, axis=1)

# Display the map
folium.LayerControl().add_to(mymap)
mymap.save(OUTPUT_PATH / "map.html")


"""
Trip generation
"""	
mobsim.trip_generation(
    share_active = 0.76, 
    share_unemployed = 0.227, 
    share_home_office = 0.0, 
    mode_split_car = 1.0, 
    car_occupancy = 1.0, 
    mode_split_motorbike = 0.0,
    motorbike_occupancy = 1.0
)

print(mobsim.mobility_zones)


df = mobsim.mobility_zones 

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
        tooltip=f'Commuters: {row.n_commuters} - Car trips: {row.n_car_trips} - Motorbike trips: {row.n_motorbike_trips} - Public trips: {row.n_motorbike_trips}'
    ).add_to(m)

# Display the map
m.save('n_commuters.html')

