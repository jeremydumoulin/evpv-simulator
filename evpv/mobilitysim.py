# coding: utf-8

""" 
MobilitySim 
Simulates the daily travel demand for different road-based transport modes (car, motorbike) for various
mobility chains specified by the user (home-work-home, home-school-home, etc)
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import json
import os
import rasterio
from pathlib import Path
from shapely.geometry import shape, LineString, Point, Polygon, box, MultiPoint
from shapely.ops import transform, nearest_points, snap
import pyproj
from pyproj import Geod
from dotenv import load_dotenv
from geopy.distance import geodesic
import osmnx as ox

from evpv import helpers as hlp

load_dotenv() # take environment variables from .env

INPUT_PATH = Path( str(os.getenv("INPUT_PATH")) )
OUTPUT_PATH = Path( str(os.getenv("OUTPUT_PATH")) )

class MobilitySim:

    #######################################
    ############# ATTRIBUTES ##############
    #######################################
    
    # Simulation are settings 

    target_area_shapefile = None
    buffer_distance = .0
    centroid_coords = list()
    simulation_bbox = list()

    # Input data

    population_density = None
    road_network = None

    #######################################
    ############### METHODS ###############
    #######################################
    
    ############# Constructor #############
    ####################################### 

    def __init__(self, target_area_shapefile, population_density, buffer_distance = .0):
        print(f"INFO \t Initializing a new MobilitySim object.")
    
        self.set_target_area_shapefile(target_area_shapefile)
        self.set_buffer_distance(buffer_distance)
        self.set_centroid_coords()
        self.set_simulation_bbox()

        self.set_population_density(population_density)
        self.set_road_network()
        
        print("INFO \t MobilitySim object created.")
        print("\t -")


    ############# Setters #############
    ###################################

    def set_target_area_shapefile(self, path):
        """ Setter for the target_area_shapefile attribute.
        """
        try:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"ERROR \t The shapefile at {path} does not exist.")

            # Load the GeoJSON file
            with open(path, 'r') as f:
                geojson_data = json.load(f)

            self.target_area_shapefile = geojson_data
        except FileNotFoundError as e:
            print(e)

    def set_buffer_distance(self, buffer_distance):
        """ Setter for buffer_distance attribute.
        Converts the value into a float
        """
        try:       
            buffer_distance = float(buffer_distance)
        except Exception as e:
            print(f"ERROR \t Impossible to convert the specified ev consumption into a float. - {e}")
        else:            
            self.buffer_distance = buffer_distance

    def set_centroid_coords(self):
        geometry = self.target_area_shapefile['features'][0]['geometry']

        # Create a shapely shape
        shapely_shape = shape(geometry)
        
        # Get the centroid of the shape
        centroid = shapely_shape.centroid
        
        # Return the coordinates of the centroid
        self.centroid_coords = centroid.y, centroid.x

    def set_simulation_bbox(self):
        geometry = self.target_area_shapefile['features'][0]['geometry']
        margin_km = self.buffer_distance

        # Create a shapely shape
        shapely_shape = shape(geometry)
        
        # Get the bounding box of the shape
        minx, miny, maxx, maxy = shapely_shape.bounds

        # Calculate the new bounding box by extending each side by the given margin in kilometers
        # Extend the minimum y (south)
        miny = geodesic(kilometers=margin_km).destination((miny, minx), 180).latitude
        # Extend the maximum y (north)
        maxy = geodesic(kilometers=margin_km).destination((maxy, minx), 0).latitude
        # Extend the minimum x (west)
        minx = geodesic(kilometers=margin_km).destination((miny, minx), 270).longitude
        # Extend the maximum x (east)
        maxx = geodesic(kilometers=margin_km).destination((miny, maxx), 90).longitude    

        # Return the coordinates of the centroid
        self.simulation_bbox = minx, miny, maxx, maxy

    def set_population_density(self, path):
        """ Setter for the population_density attribute.
        """
        try:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"ERROR \t The population density at {path} does not exist.")

            with rasterio.open(path) as dataset:
                # Read the first band
                band1 = dataset.read(1)

                hlp.crop_raster(path, self.simulation_bbox, OUTPUT_PATH / "population_density_cropped.tiff")

            self.population_density = OUTPUT_PATH / "population_density_cropped.tiff"

        except FileNotFoundError as e:
            print(e)

    def set_road_network(self):
        """ Setter for the road_network attribute.
        """
        # Convert to the format required by osmnx: north, south, east, west
        minx, miny, maxx, maxy = self.simulation_bbox
        north, south, east, west = maxy, miny, maxx, minx

        graphml_file = OUTPUT_PATH / "road_network.graphml"

        # Define the filter string to keep only motorways and primary roads
        filter_string = '["highway"!~"^(service|track|residential)$"]'

        # Check if the GraphML file exists
        if os.path.exists(graphml_file):
            # Load the graph from the GraphML file
            G = ox.load_graphml(graphml_file)
            
            # Extract the bounding box from the loaded graph
            loaded_north, loaded_south, loaded_east, loaded_west = hlp.get_graph_bbox(G)

            # Round to 4 decimal places to ignore small differences
            loaded_bbox = (round(loaded_north, 4), round(loaded_south, 4),
                   round(loaded_east, 4), round(loaded_west, 4))

            bbox = (round(north, 4), round(south, 4),
                   round(east, 4), round(west, 4))
            
            # Compare the bounding boxes
            if bbox == loaded_bbox:
                print("Found a graphml file with road network. Reusing data.")
            else:
                print("Found a graphml file with road network but the bounding box does not match. Downloading new data.")
                G = ox.graph_from_bbox(north, south, east, west, network_type='drive', custom_filter=filter_string)
                ox.save_graphml(G, graphml_file)

        else:
            # Download and extract the road network within the bounding box
            G = ox.graph_from_bbox(bbox = (north, south, east, west), network_type='drive', custom_filter=filter_string)

            # Save the graph to a GraphML file
            ox.save_graphml(G, graphml_file)

        self.road_network = G