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
from geopy.distance import geodesic, distance
import osmnx as ox
import networkx as nx
import openrouteservice
import requests
import time
import math
import csv

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
    centroid_coords = list()
    simulation_bbox = list()
    n_subdivisions = 0

    taz_width = .0
    taz_height = .0
    simulation_area_width = .0
    simulation_area_height = .0
    n_subdivisions = 0

    # Raw input data

    population_density = None
    destinations = None

    # Traffic analysis zones and associated properties 
    traffic_zones = pd.DataFrame()

    # Origin-desitnation flows 
    flows = pd.DataFrame()


    #######################################
    ############### METHODS ###############
    #######################################
    
    ############# Constructor #############
    ####################################### 

    def __init__(self, 
        target_area_shapefile, 
        population_density, 
        destinations,
        simulation_area_extension_km = 0,
        taz_target_width_km = 3,
        percentage_population_to_ignore = .0):
        
        print("---")

        print(f"INFO \t MobilitySim object initialisation...")
    
        self.set_target_area_shapefile(target_area_shapefile)

        self.set_centroid_coords()
        self.set_simulation_bbox(simulation_area_extension_km)

        self.set_simulation_area_width()
        self.set_simulation_area_height()

        self.set_taz_width(taz_target_width_km)
        self.set_n_subdivisions(self.simulation_area_width / self.taz_width)  
        self.set_taz_height()              

        self.set_population_density(population_density)
        self.set_destinations(destinations)

        self.set_traffic_zones(percentage_population_to_ignore)
        
        print(f"INFO \t MobilitySim object created:")
        print(f" \t - Simulation area - Width: {self.simulation_area_width} km | Height: {self.simulation_area_height} km")
        print(f" \t - TAZ - Number: {len(self.traffic_zones)} | Width: {self.taz_width} km | Height: {self.taz_height} km ")
        print(f" \t - Population: {self.traffic_zones['population'].sum()}")
        print(f" \t - Destinations: {self.traffic_zones['destinations'].sum()}")
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

    def set_simulation_bbox(self, simulation_area_extension_km):
        """ Calculate the bounding box for simulation from ORS isochrones
        """ 
        print(f"INFO \t Extending the simulation bbox by {simulation_area_extension_km} km")     

        # Calculate the bounding box of the target area
        geometry = self.target_area_shapefile['features'][0]['geometry']

        # Create a shapely shape
        shapely_shape = shape(geometry)

        #Get the bounding box of the shapefile defininf the target area
        minx, miny, maxx, maxy = shapely_shape.bounds

        # Calculate the new boundaries by extending them
        def extend_bbox(minx, miny, maxx, maxy, km_extension):
            # Extend minx and maxx by the km_extension in the longitudinal direction
            left_point = geodesic(kilometers=km_extension).destination((miny, minx), 270)
            right_point = geodesic(kilometers=km_extension).destination((maxy, maxx), 90)
            
            # Extend miny and maxy by the km_extension in the latitudinal direction
            bottom_point = geodesic(kilometers=km_extension).destination((miny, minx), 180)
            top_point = geodesic(kilometers=km_extension).destination((maxy, maxx), 0)
            
            new_minx = left_point.longitude
            new_maxx = right_point.longitude
            new_miny = bottom_point.latitude
            new_maxy = top_point.latitude
            
            return new_minx, new_miny, new_maxx, new_maxy

        new_minx, new_miny, new_maxx, new_maxy = extend_bbox(minx, miny, maxx, maxy, simulation_area_extension_km)

        # Return the coordinates of the centroid
        self.simulation_bbox = new_minx, new_miny, new_maxx, new_maxy

    def set_simulation_area_width(self):
        minx, miny, maxx, maxy = self.simulation_bbox

        self.simulation_area_width = geodesic((minx, miny), (maxx, miny)).kilometers

    def set_simulation_area_height(self):
        minx, miny, maxx, maxy = self.simulation_bbox

        self.simulation_area_height = geodesic((minx, miny), (minx, maxy)).kilometers

    def set_taz_width(self, taz_target_width_km):
        # Compute the number of integer segments close to the target width
        n = round(self.simulation_area_width / taz_target_width_km)

        # Calculate the actual segment length
        l = self.simulation_area_width / n

        self.taz_width = l

    def set_taz_height(self):
        minx, miny, maxx, maxy = self.simulation_bbox

        self.taz_height = self.simulation_area_height / self.n_subdivisions

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

    def set_destinations(self, csv_file_path):
            """Setter for the destinations using a CSV file with 4 cols: name, latitude, longitude, weigth
            """
            print(f"INFO \t Appending the destinations and weights from CSV file.")

            center_points = []

            with open(csv_file_path, mode='r') as csv_file:
                csv_reader = csv.DictReader(csv_file)
                for row in csv_reader:
                    name = row['name']
                    latitude = float(row['latitude'])
                    longitude = float(row['longitude'])
                    weight = int(row['weight'])

                    if weight < 1:
                        print(f"ALERT \t Skipping {name} due to non-positive weight: {weight}")
                        continue
                    
                    for _ in range(weight):
                        center_points.append((longitude, latitude))

            self.destinations = center_points            


    def set_traffic_zones(self, percentage_population_to_ignore = .0):
        """ Setter for the traffic_zones attribute.
        """

        print(f"INFO \t Setting up traffic analysis zones (TAZs) and associated features")

        # Split the area into n x n zones 

        minx, miny, maxx, maxy = self.simulation_bbox

        width_bbox = self.simulation_area_width
        size_unit_cell = self.taz_width
        num_squares = self.n_subdivisions
    
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
                center_lon = miny + (i + 0.5) * height
                center_lat = minx + (j + 0.5) * width

                # 2. Bounding box 

                lower_left_x = minx + i * width
                lower_left_y = miny + j * height
                upper_right_x = lower_left_x + width
                upper_right_y = lower_left_y + height

                bbox_geom = box(lower_left_x, lower_left_y, upper_right_x, upper_right_y)

                # 3. Population within the bounding box

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

                # 4. Number of destinations

                # Convert the list of center points to shapely Point objects
                points = [Point(lon, lat) for lon, lat in self.destinations]

                # Count how many points are within the bounding box
                points_within_bbox = [point for point in points if point.within(bbox_geom)]
                n_destinations = len(points_within_bbox)

                # 5. Check if the TAZ is inside the target area
                
                shapefile_geometry = shape(self.target_area_shapefile['features'][0]['geometry'])
                center_point = Point((center_lat, center_lon))

                # Check if the point is within the MultiPolygon
                is_within = shapefile_geometry.contains(center_point)

                # 5. Append everything

                grid_data.append({'id': zone_id, 'geometric_center': (center_lat, center_lon), 'bbox': bbox_geom, 'population': total_population, 'destinations': n_destinations, 'is_within_target_area': is_within})

        df = pd.DataFrame(grid_data)

        # Delete the sparsely TAZs, such that the sum of the less populated is below a threshold
        if percentage_population_to_ignore > .0:
            print(f"INFO \t Deleting sparsely populated TAZs")

            # Calculate the total population
            total_population = df['population'].sum()

            # Calculate the population limit based on the percentage
            population_limit = (percentage_population_to_ignore / 100) * total_population

            # Sort by population in ascending order
            df = df.sort_values(by='population')

            # Calculate the cumulative sum of populations
            df['cumulative_population'] = df['population'].cumsum()

            # Identify the rows to remove
            rows_to_remove = df[df['cumulative_population'] <= population_limit]

            # Drop these rows from the original DataFrame
            df = df.drop(rows_to_remove.index)

            # Optionally, remove the cumulative_population column
            df = df.drop(columns=['cumulative_population'])

        self.traffic_zones = pd.DataFrame(df)


    ############# Trip Generation #############
    ###########################################

    def trip_generation(self, n_trips_per_inhabitant):
        print(f"INFO \t Starting trip generation")

        if n_trips_per_inhabitant <= .0:
            print(f"ERROR \t Trips per inhabitant must be greater than 0")
            return

        # Load the traffic_zones dataframe

        df = self.traffic_zones

        # Calculate the number of trips and append them to the df
        df['n_outflows'] = df['population'].apply( lambda x: int(x * n_trips_per_inhabitant) )

        self.traffic_zones = df

        print(f"INFO \t Trip generation done. Data has been appended to the traffic_zones attribute. Trip distribution required!")

    ############ Trip Distribution ############
    ###########################################

    def trip_distribution(self, model, attraction_feature = "population", cost_feature = "distance_road", batch_size = 49):
        print(f"INFO \t Starting trip distribution")

        ############ Get TAZ data ############

        df = self.traffic_zones

        # Check if trip generation has been performed
        if not 'n_outflows' in df.columns:
            print(f"ERROR \t Traffic Analysis Zones do not contain the number of trips: trip generation must be performed before trip distribution.")
            return

        ############ Get ORS data ############

        print(f"INFO \t Getting ORS data")

        # Extract coordinates of the centroid
        coordinates = [list(coord) for coord in df['geometric_center']] 
        num_coordinates = len(coordinates)

        # Check if the number of coordinates exceeds the batch size
        if num_coordinates > batch_size:
            print(f"ALERT \t {num_coordinates} origins/destinations: this number is greater than the batch size set at {batch_size}. Multiple ORS requests are needed.")

        # Initialize ORS client
        client = openrouteservice.Client(key=str(os.getenv("ORS_KEY")))  # Replace with your ORS API key

        # Split the coordinates into manageable batches
        coordinate_batches = [coordinates[i:i+batch_size] for i in range(0, len(coordinates), batch_size)]

        # Initialize the full matrices
        durations = np.zeros((num_coordinates, num_coordinates))
        distances = np.zeros((num_coordinates, num_coordinates))

        # Make multiple requests to the ORS API for each batch
        for i, batch in enumerate(coordinate_batches):
            print(f"INFO \t Sending ORS request for {len(batch)} origins/destinations.")

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

        # Prepare lists to hold the data
        origin_ids = []
        destination_ids = []
        flows = []
        travel_times = []
        travel_distances = []
        travel_distances_euclidian = []

        # Populate the data based on the matrix response
        ors_errors = 0
        for i, origin_id in enumerate(df['id']):
            for j, destination_id in enumerate(df['id']):
                origin_ids.append(origin_id)
                destination_ids.append(destination_id)

                flows.append(0.0)  # Placeholded for the flow for all origin-destination pairs                

                # Euclidian travel distance 
                point1 = df.loc[df['id'] == origin_id, 'geometric_center'].iloc[0]
                point2 = df.loc[df['id'] == destination_id, 'geometric_center'].iloc[0]

                euclidian_distance = geodesic(point1, point2).kilometers

                travel_distances_euclidian.append(euclidian_distance)

                # Check if ors errors. Display only once
                if (math.isnan(distances[i][j]) or distances[i][j] == .0) and (origin_id != destination_id):
                    travel_distances.append(euclidian_distance)
                    travel_times.append(euclidian_distance / 30 * 60)

                    ors_errors = ors_errors + 1
                        
                else:
                    travel_distances.append(distances[i][j] / 1000)  # Convert meters to kilometer
                    travel_times.append(durations[i][j] / 60)  # Convert seconds to minutes

        if ors_errors != 0:
            print(f"ALERT \t ORS was unable to calculate distance or resolve {ors_errors} routes. Using euclidian distance instead and a travel speed of 30 km/h. This could affect the model reliability!")

        # Create the resulting DataFrame
        flow_data = {
            'Origin': origin_ids,
            'Destination': destination_ids,
            'Flow': flows,
            'Travel Time (min)': travel_times,
            'Travel Distance (km)': travel_distances,
            'Centroid Distance (km)': travel_distances_euclidian
        }

        flows_df = pd.DataFrame(flow_data)

        ############ Remove rows where 'Origin' is equal to 'Destination' ############

        # This is to ensure that spatial interaction model holds 

        flows_df = flows_df[flows_df['Origin'] != flows_df['Destination']]

        ############ Apply spatial interaction model ############

        print(f"INFO \t Applying spatial interaction model")

        # Iterate over each origin and apply the model
        for origin in flows_df['Origin'].unique():
            # Filter rows for the current origin
            origin_rows = flows_df[flows_df['Origin'] == origin]
            origin_id = origin_rows.iloc[0]['Origin']

            # SIM input: outgoing trips
            n_outflows = self.traffic_zones.loc[self.traffic_zones['id'] == origin_id , 'n_outflows'].values[0]

            # SIM input: attraction
            if attraction_feature == 'population':
                att_origin = self.traffic_zones.loc[self.traffic_zones['id'] == origin_id , 'population'].values[0]
                dest_att_list = origin_rows['Destination'].apply(lambda x: self.traffic_zones.loc[self.traffic_zones['id'] == x, 'population'].values[0]).tolist()          
            elif attraction_feature == 'destinations':
                att_origin = self.traffic_zones.loc[self.traffic_zones['id'] == origin_id , 'destinations'].values[0]
                dest_att_list = origin_rows['Destination'].apply(lambda x: self.traffic_zones.loc[self.traffic_zones['id'] == x, 'destinations'].values[0]).tolist()
            else:
                print(f"ERROR \t Attraction feature is unknown.")
                return

            # SIM input: cost
            if cost_feature == 'distance_road':
                cost_list = origin_rows['Travel Distance (km)'].tolist() # Extract the "Travel Distance (km)" column into a list
            elif cost_feature == 'time_road':
                cost_list = origin_rows['Travel Time (min)'].tolist() # Extract the "Travel Time (min)" column into a list
            elif cost_feature == 'distance_centroid':
                cost_list = origin_rows['Centroid Distance (km)'].tolist() # Extract the "Centroid Distance (km)" column into a list
            else:
                print(f"ERROR \t Cost feature is unknown.")
                return        

            # Calculate the flows depending on the model 
            if model == 'gravity_power_1':
                flows = hlp.prod_constrained_gravity_power(
                    origin_n_trips = n_outflows,
                    dest_attractivity_list = dest_att_list,                
                    cost_list = cost_list, 
                    gamma = 1)
            elif model == 'gravity_exp_1':
                flows = hlp.prod_constrained_gravity_exp(
                    origin_n_trips = n_outflows,
                    dest_attractivity_list = dest_att_list,                
                    cost_list = cost_list, 
                    beta = 1)
            elif model == 'gravity_exp_01':
                flows = hlp.prod_constrained_gravity_exp(
                    origin_n_trips = n_outflows,
                    dest_attractivity_list = dest_att_list,                
                    cost_list = cost_list, 
                    beta = 0.1)
            elif model == 'gravity_exp_016':
                flows = hlp.prod_constrained_gravity_exp(
                    origin_n_trips = n_outflows,
                    dest_attractivity_list = dest_att_list,                
                    cost_list = cost_list, 
                    beta = 0.16)
            elif model == 'gravity_exp_scaled':
                flows = hlp.prod_constrained_gravity_exp(
                    origin_n_trips = n_outflows,
                    dest_attractivity_list = dest_att_list,                
                    cost_list = cost_list, 
                    beta = 0.3 * (self.taz_width*self.taz_width)**(-0.18) )
            elif model == 'radiation':
                flows = hlp.prod_constrained_radiation(
                    origin_n_trips = n_outflows,
                    origin_attractivity = att_origin,
                    dest_attractivity_list = dest_att_list,                
                    cost_list = cost_list)
            elif model == 'radius_6km':
                flows = hlp.prod_constrained_radius(
                    origin_n_trips = n_outflows,
                    dest_attractivity_list = dest_att_list,                
                    cost_list = cost_list,
                    radius = 6)
            else:
                print(f"ERROR \t Spatial interaction model '{model}' is unknown.")
                return   

            # Update the flows column where row_id equals 1
            flows_df.loc[flows_df['Origin'] == origin_id, 'Flow'] = flows[:len(flows_df.loc[flows_df['Origin'] == origin_id])]

        ############ Append flow data ############

        self.flows = flows_df

    ################# Routing #################
    ###########################################

    def allocate_routes(self):
        print(f"INFO \t Allocation of ORS routes to origin-destination pairs (routing)")

        flows = self.flows
        taz = self.traffic_zones

        # Add a new column for the route geometry
        flows['Geometry'] = None

        # Initialize ORS client
        client = openrouteservice.Client(key=str(os.getenv("ORS_KEY")))  # Replace with your ORS API key

        # Create a dictionary to store previously calculated routes (avoids recalculating when origin and destination are swapped)
        route_cache = {}

        i = 0
        for index, row in flows.iterrows():
            i = i+1

            print(f"INFO \t Allocation: {i} out of {len(flows)}", end="\r")

            origin_id = row['Origin']
            destination_id = row['Destination']
            flow = row['Flow']
            
            # Create a unique key for each origin-destination pair
            route_key = tuple(sorted([origin_id, destination_id]))
            
            if route_key in route_cache:
                # If the route has been calculated before, use the cached geometry
                geometry = route_cache[route_key]
                # Check if we need to reverse the LineString
                if (origin_id, destination_id) != route_key:
                    geometry = LineString(geometry.coords[::-1])
            else:
                # Get the coordinates from taz
                origin_coords = taz.loc[taz['id'] == origin_id, 'geometric_center'].values[0]
                destination_coords = taz.loc[taz['id'] == destination_id, 'geometric_center'].values[0]
                
                origin_lon, origin_lat = origin_coords
                destination_lon, destination_lat = destination_coords 
                
                # Get the route from ORS
                try:
                    route = client.directions(
                        coordinates=[origin_coords, destination_coords],
                        profile='driving-car',
                        format='geojson'
                    )
                    geometry = LineString(route['features'][0]['geometry']['coordinates'])
                    # Cache the route
                    route_cache[route_key] = geometry
                except Exception as e:
                    print(f"ERROR \t An error occurred in route calculation - Geometry is set to None")
                    geometry = None
            
            flows.at[index, 'Geometry'] = geometry
            
            # Adding a sleep time to avoid hitting the rate limit
            time.sleep(1.5) 


    ############### Deprecated ################
    ###########################################

    # def set_road_network(self, road_network_filter_string):
    #     """ Setter for the road_network attribute.
    #     """     

    #     # Convert to the format required by osmnx: north, south, east, west
    #     minx, miny, maxx, maxy = self.simulation_bbox
    #     north, south, east, west = maxy, miny, maxx, minx

    #     graphml_file = OUTPUT_PATH / "road_network.graphml"

    #     # Define the filter string to keep only motorways and primary roads
    #     filter_string = road_network_filter_string

    #     print(f"INFO \t Getting the road network from OSM. Applied filter: {filter_string}")

    #     ox.settings.use_cache=False
    #     #ox.config(use_cache=False)

    #     # Check if the GraphML file exists
    #     if os.path.exists(graphml_file):
    #         # Load the graph from the GraphML file
    #         G = ox.load_graphml(graphml_file)
            
    #         # Extract the bounding box from the loaded graph
    #         loaded_north, loaded_south, loaded_east, loaded_west = hlp.get_graph_bbox(G)

    #         # Round to decimal places to ignore small differences
    #         decimals = 2

    #         loaded_bbox = (round(loaded_north, decimals), round(loaded_south, decimals),
    #                round(loaded_east, decimals), round(loaded_west, decimals))

    #         bbox = (round(north, decimals), round(south, decimals),
    #                round(east, decimals), round(west, decimals))
            
    #         # Compare the bounding boxes
    #         if bbox == loaded_bbox:
    #             print(f"\t -> Found a graphml file with road network. Reusing existing data.")
    #         else:
    #             print(f"\t -> Found a graphml file with road network but the bounding box does not match. Downloading new data.")
    #             #G = ox.graph_from_bbox(north, south, east, west, network_type='drive')
    #             G = ox.graph_from_bbox(bbox = (north, south, east, west), network_type='drive', custom_filter=filter_string)

    #             ox.save_graphml(G, graphml_file)

    #     else:
    #         # Download and extract the road network within the bounding box
    #         G = ox.graph_from_bbox(bbox = (north, south, east, west), network_type='all', custom_filter=filter_string)

    #         # Save the graph to a GraphML file
    #         ox.save_graphml(G, graphml_file)

    #     self.road_network = G
        