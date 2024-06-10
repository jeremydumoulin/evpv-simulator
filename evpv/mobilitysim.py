# coding: utf-8

""" 
MobilitySim 

A class to simulate the daily travel demand for different road-based transport modes (car, motorbike) for various
mobility chains specified by the user (home-work-home, home-school-home, etc)
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
from geopy.distance import geodesic
import osmnx as ox
import openrouteservice
import time

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
    n_subdivisions = 0

    subdivision_size = .0
    simulation_area_size = .0

    # Raw input data

    population_density = None
    road_network = None
    workplaces = None

    # Traffic analysis zones and associated properties 
    traffic_zones = pd.DataFrame()

    # Origin-desitnation flows by commuting type 
    flows_car = pd.DataFrame()
    flows_motorbike = pd.DataFrame()
    flows_public = pd.DataFrame()

    #######################################
    ############### METHODS ###############
    #######################################
    
    ############# Constructor #############
    ####################################### 

    def __init__(self, target_area_shapefile, 
        population_density, 
        buffer_distance = .0, 
        n_subdivisions = 10, 
        road_network_filter_string = '["highway"!~"^(service|track|residential)$"]',
        workplaces_tags = {
            "building": ["industrial", "office"],
            "company": [],
            "landuse": ["industrial"],
            "industrial": [],
            "office": ["company", "government"],
            "amenity": ["university", "research_institute", "conference_centre", "bank", "hospital", "townhall", "police", "fire_station", "post_office", "post_depot"]
        }):
        
        print("---")

        print(f"INFO \t MobilitySim object initialisation with {n_subdivisions}x{n_subdivisions} TAZs and {buffer_distance} km buffer distance")
    
        self.set_target_area_shapefile(target_area_shapefile)
        self.set_buffer_distance(buffer_distance)
        self.set_n_subdivisions(n_subdivisions)
        self.set_centroid_coords()
        self.set_simulation_bbox()

        self.set_simulation_area_size()
        self.set_subdivision_size()        

        self.set_population_density(population_density)
        self.set_road_network(road_network_filter_string)
        self.set_workplaces(workplaces_tags)

        self.set_traffic_zones(n_subdivisions)
        
        print(f"INFO \t MobilitySim object created:")
        print(f" \t - Simulation area bbox length: {self.simulation_area_size} km")
        print(f" \t - TAZ length: {self.subdivision_size} km")
        print(f" \t - Population: {self.traffic_zones['population'].sum()}")
        print(f" \t - Workplaces: {self.traffic_zones['workplaces'].sum()}")
        print("---")

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
            print(f"ERROR \t Impossible to convert the specified buffer distance into a float. - {e}")
        else:            
            self.buffer_distance = buffer_distance

    def set_n_subdivisions(self, n_subdivisions):
        """ Setter for n_subdivision attribute.
        Converts the value into an int
        """
        try:       
            n_subdivisions = int(n_subdivisions)
        except Exception as e:
            print(f"ERROR \t Impossible to convert the specified subdivisions into a int. - {e}")
        else:            
            self.n_subdivisions = n_subdivisions

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

    def set_simulation_area_size(self):
        minx, miny, maxx, maxy = self.simulation_bbox

        self.simulation_area_size = geodesic((miny, minx), (miny, maxx)).kilometers

    def set_subdivision_size(self):
        self.subdivision_size = self.simulation_area_size / self.n_subdivisions

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

    def set_road_network(self, road_network_filter_string):
        """ Setter for the road_network attribute.
        """     

        # Convert to the format required by osmnx: north, south, east, west
        minx, miny, maxx, maxy = self.simulation_bbox
        north, south, east, west = maxy, miny, maxx, minx

        graphml_file = OUTPUT_PATH / "road_network.graphml"

        # Define the filter string to keep only motorways and primary roads
        filter_string = road_network_filter_string

        print(f"INFO \t Getting the road network from OSM. Applied filter: {filter_string}")

        ox.settings.use_cache=False
        #ox.config(use_cache=False)

        # Check if the GraphML file exists
        if os.path.exists(graphml_file):
            # Load the graph from the GraphML file
            G = ox.load_graphml(graphml_file)
            
            # Extract the bounding box from the loaded graph
            loaded_north, loaded_south, loaded_east, loaded_west = hlp.get_graph_bbox(G)

            # Round to decimal places to ignore small differences
            decimals = 2

            loaded_bbox = (round(loaded_north, decimals), round(loaded_south, decimals),
                   round(loaded_east, decimals), round(loaded_west, decimals))

            bbox = (round(north, decimals), round(south, decimals),
                   round(east, decimals), round(west, decimals))
            
            # Compare the bounding boxes
            if bbox == loaded_bbox:
                print(f"\t -> Found a graphml file with road network. Reusing existing data.")
            else:
                print(f"\t -> Found a graphml file with road network but the bounding box does not match. Downloading new data.")
                #G = ox.graph_from_bbox(north, south, east, west, network_type='drive')
                G = ox.graph_from_bbox(north, south, east, west, network_type='drive', custom_filter=filter_string)

                ox.save_graphml(G, graphml_file)

        else:
            # Download and extract the road network within the bounding box
            G = ox.graph_from_bbox(north, south, east, west, network_type='drive', custom_filter=filter_string)

            # Save the graph to a GraphML file
            ox.save_graphml(G, graphml_file)

        self.road_network = G

    def set_workplaces(self, workplaces_tags):
        """ Setter for the workplaces
        """       

        # Extract the coordinates of the bounding box vertices
        minx, miny, maxx, maxy = self.simulation_bbox
        bbox_coords = [maxy, miny, maxx, minx]

        # Amenities to extract
        tags = workplaces_tags

        print(f"INFO \t Getting the workplaces from OSM. Tags: {tags}")

        mypois = ox.features.features_from_bbox(bbox=bbox_coords, tags=tags) # the table

        # Convert ways into nodes and get the coordinates of the center point - Do not store the relations
        center_points = []
        for index, row in mypois.iterrows():
            if index[0] == 'way':
                shapefile = row['geometry']
                center_point = shapefile.centroid
                center_points.append((center_point.x, center_point.y))
            if index[0] == 'node':
                center_points.append((row['geometry'].x, row['geometry'].y))
            else:
                break

        self.workplaces = center_points

    def set_traffic_zones(self, num_squares):
        """ Setter for the traffic_zones attribute.
        """

        print(f"INFO \t Setting up traffic analysis zones (TAZs) and associated features")

        # Split the area into num_squares x num_squares zones 

        minx, miny, maxx, maxy = self.simulation_bbox
    
        # Calculate the width and height of each zone
        width = (maxx - minx) / num_squares
        height = (maxy - miny) / num_squares
        
        grid_data = []
        
        # Loop to create grid and calculate center of each square
        for i in range(num_squares):
            for j in range(num_squares):
                # 0. ID 
                zone_id = f"{i}_{j}"

                # 1. Latitude and longitude of the geometric center 
                center_lat = minx + (i + 0.5) * width
                center_lon = miny + (j + 0.5) * height

                # 2. Nearest node in the road network

                nearest_node = ox.distance.nearest_nodes(self.road_network, center_lat, center_lon)

                # 3. Bounding box 

                lower_left_x = minx + i * width
                lower_left_y = miny + j * height
                upper_right_x = lower_left_x + width
                upper_right_y = lower_left_y + height

                bbox_geom = box(lower_left_x, lower_left_y, upper_right_x, upper_right_y)

                # 4. Population within the bounding box

                # GeoDataFrame
                bbox_gdf = gpd.GeoDataFrame({'geometry': [bbox_geom]}, crs="EPSG:4326")

                # Path to the population raster file
                population_raster_path = self.population_density

                # Read the population raster
                with rasterio.open(population_raster_path) as src:
                    # Clip the raster using the bounding box
                    out_image, out_transform = mask(src, [bbox_gdf.geometry.values[0]], crop=True)
                    out_meta = src.meta

                # Calculate the total population within the bounding box
                total_population = np.sum(out_image[out_image > 0])  # assuming no data values are <= 0

                # 5. Number of workplaces

                # Convert the list of center points to shapely Point objects
                points = [Point(lon, lat) for lon, lat in self.workplaces]

                # Count how many points are within the bounding box
                points_within_bbox = [point for point in points if point.within(bbox_geom)]
                n_workplaces = len(points_within_bbox)

                # 6. Append everything

                grid_data.append({'id': zone_id, 'geometric_center': (center_lat, center_lon), 'nearest_node': (self.road_network.nodes[nearest_node]['x'], self.road_network.nodes[nearest_node]['y']), 'bbox': bbox_geom, 'population': total_population, 'workplaces': n_workplaces})

        self.traffic_zones = pd.DataFrame(grid_data)


    ############# Trip Generation #############
    ###########################################

    def trip_generation(self, share_active, share_unemployed, share_home_office, mode_share, vehicle_occupancy):
        print(f"INFO \t Generating the number of trips from each TAZ")

        # Check the values 

        params = [share_active, share_unemployed, share_home_office, mode_share]
    
        if not all(0.0 <= param <= 1.0 for param in params):
            print(f"ERROR \t All parameters must be between 0 and 1.")
            return

        # Load the traffic_zones dataframe

        df = self.traffic_zones

        # Calculate the number of trips and append them to the df
        df['n_commuters'] = df['population'].apply( lambda x: int(x * share_active * (1 - share_unemployed) * (1 - share_home_office)) )
        df['n_trips'] = df['n_commuters'].apply( lambda x: int(x * mode_share / vehicle_occupancy) )

        self.traffic_zones = df

        print(f"INFO \t Trip generation done. Data has been appended to the traffic_zones attribute.")

    ############ Trip Distribution ############
    ###########################################

    def trip_distribution(self, mode, model = "radiation", min_distance = 10, batch_size = 49):
        df = self.traffic_zones

        # Extract coordinates
        coordinates = [list(coord) for coord in df['nearest_node']]
        num_coordinates = len(coordinates)

        if num_coordinates > batch_size:
            print(f"ALERT \t {num_coordinates} origin-destination pairs: this number is greater than the batch size set at {batch_size}. Multiple ORS requests are needed.")

        # Initialize ORS client
        client = openrouteservice.Client(key=str(os.getenv("ORS_KEY")))  # Replace with your ORS API key

        # Split the coordinates into manageable batches
        coordinate_batches = [coordinates[i:i+batch_size] for i in range(0, len(coordinates), batch_size)]

        # Initialize the full matrices
        durations = np.zeros((num_coordinates, num_coordinates))
        distances = np.zeros((num_coordinates, num_coordinates))

        # Make multiple requests to the ORS API for each batch
        for i, batch in enumerate(coordinate_batches):
            print(f"ALERT \t Sending ORS request for {len(batch)} origin-destination pairs.")

            batch_matrix = client.distance_matrix(
                locations=batch,
                profile='driving-car',
                metrics=['duration', 'distance'],
                resolve_locations=True
            )
            
            # Extract travel times and distances from the response
            batch_durations = batch_matrix['durations']
            batch_distances = batch_matrix['distances']
            
            # Determine the range of indices for this batch
            batch_start = i * batch_size
            batch_end = batch_start + len(batch)
            
            # Populate the corresponding section of the full matrices
            durations[batch_start:batch_end, batch_start:batch_end] = batch_durations
            distances[batch_start:batch_end, batch_start:batch_end] = batch_distances
            
            time.sleep(2)  # Introduce a delay between requests to avoid rate limiting

        # Define constants for distance and time within two points located in the same zone
        minx, miny, maxx, maxy = self.simulation_bbox

        a = geodesic((miny, minx), (miny, maxx)).kilometers / self.n_subdivisions # size of the square in km
        d_avg = a * 0.52  # average distance between two randomy distributed points in a square
        speed_kmh = 20  # speed in km/h
        t = d_avg / speed_kmh * 60 # Travel time in minutes

        # Prepare lists to hold the data
        origin_ids = []
        destination_ids = []
        flows = []
        travel_times = []
        travel_distances = []

        # Populate the data based on the matrix response
        for i, origin_id in enumerate(df['id']):
            for j, destination_id in enumerate(df['id']):
                origin_ids.append(origin_id)
                destination_ids.append(destination_id)
                flows.append(0.0)  # Placeholded for the flow for all origin-destination pairs

                if origin_id == destination_id:
                    travel_distances.append(d_avg)  # average distance in the same zone
                    travel_times.append(t)  # time in minutes
                else:
                    travel_distances.append(distances[i][j] / 1000)  # Convert meters to kilometers

                    # Calculated ORS duration...
                    #travel_times.append(durations[i][j] / 60)  # Convert seconds to minutes
                    # ... or duration from average travel speed
                    travel_times.append(distances[i][j] / 1000 / speed_kmh * 60)  # time in minutes

        # Create the resulting DataFrame
        flow_data = {
            'Origin': origin_ids,
            'Destination': destination_ids,
            'Flow': flows,
            'Travel Time (min)': travel_times,
            'Travel Distance (km)': travel_distances
        }

        flows_df = pd.DataFrame(flow_data)

        # Iterate over each origin and apply the model
        for origin in flows_df['Origin'].unique():
            # Filter rows for the current origin
            origin_rows = flows_df[flows_df['Origin'] == origin]

            # Sort the DataFrame by the value of travel distance
            origin_rows = origin_rows.sort_values(by='Travel Distance (km)')
            # origin_rows = origin_rows.sort_values(by='Travel Time (min)')

            # Get the population of the origin
            origin_id = origin_rows.iloc[0]['Origin']
            pop_origin = self.traffic_zones.loc[self.traffic_zones['id'] == origin_id , 'population'].values[0]
            departures = self.traffic_zones.loc[self.traffic_zones['id'] == origin_id , 'n_car_trips'].values[0]

            # Initialize the population counter of the intervening opportunities
            pop_intervening = 0
            flow_value_sum = .0

            print(pop_origin)

            # Exclude destinations less than 5 km away
            origin_rows = origin_rows[origin_rows['Travel Distance (km)'] >= min_distance]

            # Iterate over each destination in the sorted DataFrame
            j = 0
            for index, row in origin_rows.iterrows():
                # Get the destination population and distance 
                destination_id = row['Destination']
                pop_destination = self.traffic_zones.loc[self.traffic_zones['id'] == destination_id, 'population'].values[0]

                # Compute the flows using a radiation model 
                num = pop_origin * pop_destination
                den = (pop_origin + pop_intervening) * (pop_origin + pop_destination + pop_intervening)

                flow_value = num / den

                print(flow_value)

                flow_value_sum += flow_value

                if j >= 2:
                    pop_intervening += pop_destination # Adding the intervening opportunity population, skipping the population at the origin

                j += 1

                # Update the Flow value in the original DataFrame
                flows_df.loc[index, 'Flow'] = flow_value

            # Normalisation and number of trips affectation
            pop_tot = 0
            for index, row in origin_rows.iterrows():
                flows_df.loc[index, 'Flow'] = int( flows_df.loc[index, 'Flow'] / flow_value_sum * pop_origin )

            # Calculate and print the sum of flows for the current origin
            self.flows_car = flows_df