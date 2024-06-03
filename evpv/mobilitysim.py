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

    # Raw input data

    population_density = None
    road_network = None
    workplaces = None

    # Mobility zones - features 
    mobility_zones = pd.DataFrame()

    #######################################
    ############### METHODS ###############
    #######################################
    
    ############# Constructor #############
    ####################################### 

    def __init__(self, target_area_shapefile, population_density, buffer_distance = .0, n_subdivisions = 10):
        print(f"INFO \t Initializing a new MobilitySim object.")
    
        self.set_target_area_shapefile(target_area_shapefile)
        self.set_buffer_distance(buffer_distance)
        self.set_centroid_coords()
        self.set_simulation_bbox()

        self.set_population_density(population_density)
        self.set_road_network()
        self.set_workplaces()

        self.set_mobility_zones(n_subdivisions)
        
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

        ox.settings.use_cache=False
        #ox.config(use_cache=False)

        # Check if the GraphML file exists
        if os.path.exists(graphml_file):
            # Load the graph from the GraphML file
            G = ox.load_graphml(graphml_file)
            
            # Extract the bounding box from the loaded graph
            loaded_north, loaded_south, loaded_east, loaded_west = hlp.get_graph_bbox(G)

            # Round to decimal places to ignore small differences
            decimals = 3

            loaded_bbox = (round(loaded_north, decimals), round(loaded_south, decimals),
                   round(loaded_east, decimals), round(loaded_west, decimals))

            bbox = (round(north, decimals), round(south, decimals),
                   round(east, decimals), round(west, decimals))
            
            # Compare the bounding boxes
            if bbox == loaded_bbox:
                print("Found a graphml file with road network. Reusing existing data.")
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

    def set_workplaces(self):
        """ Setter for the workplaces
        """
        # Extract the coordinates of the bounding box vertices
        minx, miny, maxx, maxy = self.simulation_bbox
        bbox_coords = [maxy, miny, maxx, minx]

        # Amenities to extract
        tags = {
            "building": ["industrial", "office"],
            "company": [],
            "landuse": ["industrial"],
            "industrial": [],
            "office": ["company", "government"],
            "amenity": ["university", "research_institute", "conference_centre", "bank", "hospital", "townhall", "police", "fire_station", "post_office", "post_depot"]
        }

        mypois = ox.features.features_from_bbox(bbox=bbox_coords, tags=tags) # the table

        # print(len(mypois))
        # mypois.head(5)

        # Convert ways into nodes and get the coordinates of the center point - Do not store the relations
        center_points = []
        for element_type, osmid in mypois.index:
            if element_type == 'way':
                shapefile = mypois.loc[(element_type, osmid), 'geometry']
                center_point = shapefile.centroid
                center_points.append((center_point.x, center_point.y))
            if element_type == 'node':
                center_points.append((mypois.loc[(element_type, osmid), 'geometry'].x, mypois.loc[(element_type, osmid), 'geometry'].y))
            else:
                break

        self.workplaces = center_points

    def set_mobility_zones(self, num_squares):
        """ Setter for the mobility_zones attribute.
        """

        # 1. Split the area into num_squares x num_squares zones 

        minx, miny, maxx, maxy = self.simulation_bbox
    
        # Calculate the width and height of each zone
        width = (maxx - minx) / num_squares
        height = (maxy - miny) / num_squares
        
        grid_data = []
        
        # Loop to create grid and calculate center of each square
        for i in range(num_squares):
            for j in range(num_squares):
                # Latitude and longitude of the geometric center 
                center_lat = minx + (i + 0.5) * width
                center_lon = miny + (j + 0.5) * height

                # Bounding box 

                lower_left_x = minx + j * width
                lower_left_y = miny + i * height
                upper_right_x = lower_left_x + width
                upper_right_y = lower_left_y + height

                bbox_geom = box(lower_left_x, lower_left_y, upper_right_x, upper_right_y)

                # Append everything

                grid_data.append({'geometric_center': (center_lat, center_lon), 'bbox': bbox_geom})

        self.mobility_zones = pd.DataFrame(grid_data)

        #print(mobility_zones)