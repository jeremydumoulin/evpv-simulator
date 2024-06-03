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

# Initialize mobility simulation
mobsim = MobilitySim(shapefile_path, population_density = population_density_path, buffer_distance = 0, n_subdivisions = 5)

# print(mobsim.centroid_coords)
# print(mobsim.simulation_bbox)



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
folium.GeoJson(gdf, name='Road Network').add_to(mymap)


# Add workplaces
# Add markers for each center point
for point in mobsim.workplaces:
    folium.Marker(location=[point[1], point[0]], popup="Center point").add_to(mymap)

# Add the subdivisions

# Function to add rectangles to the map
def add_rectangle(row):
    # Parse the WKT string to create a Polygon object
    bbox_polygon = row['bbox']
    bbox_coords = bbox_polygon.bounds
    
    # Add rectangle to map
    folium.Rectangle(
        bounds=[(bbox_coords[1], bbox_coords[0]), (bbox_coords[3], bbox_coords[2])],
        color='blue',
        fill=True,
        fill_color='blue',
        fill_opacity=0.2
    ).add_to(mymap)


# Apply the function to each row in the DataFrame
mobsim.mobility_zones.apply(add_rectangle, axis=1)


# Display the map
folium.LayerControl().add_to(mymap)
mymap.save(OUTPUT_PATH / "map.html")	


