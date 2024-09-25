# coding: utf-8

import json
import os
import rasterio
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import shape, LineString, Point, Polygon, box, MultiPoint
from shapely.ops import transform, nearest_points, snap
from geopy.distance import geodesic, distance
import openrouteservice
import time
import math
import csv
import pickle
import folium
import branca.colormap as cm

from evpv import helpers as hlp

class MobilitySim:
    """
    A class to simulate the daily travel demand for various road-based transport modes (e.g., car, motorbike) from home (origin) 
    to commuting destinations (e.g., workplaces, park-and-ride facilities, ...). 

    The model operates as follows:
    1. Traffic Zone Division: The target area is divided into traffic analysis zones.
    2. Transport Demand Modelling:
    - Trip Generation: Based on georeferenced population density and the average number of people commuting.
    - Trip Distribution: Calculated origin-destination flows from home to user-defined georeferenced destinations using a 
    spatial interaction model (such as gravity or radiation) populated with road-based distances between traffic zones (using Open Route Service). 
    An self-calibrated gravity model is also available to avoid the need for model calibration when no data is available.
        
    The class also includes some methods for post-processing and visualization.
    """

    #######################################
    ############# Constructor #############
    ####################################### 

    def __init__(self, target_area: str, population_density: str, destinations: str) -> None:
        """
        Initialize a new instance of the MobilitySim class.

        Args:
            target_area (str): Path to the GeoJSON file representing the target area.
            population_density (str): Path to the file with population density data.
            destinations (str): Path to the CSV file with destination points and their weights.
        """
        self.target_area = target_area
        self.population_density = population_density
        self.destinations = destinations

        # Transport model setup 
        self._simulation_bbox = None
        self._taz_width = 0.0

        # Transport model results
        self._traffic_zones = pd.DataFrame()
        self._flows = pd.DataFrame()

        # Track the state of the object
        self.state = 'created'  

        print(f"INFO \t New MobilitySim object created")

    #######################################
    ####### Main Getters and Setters ######
    #######################################

    # Target area
    @property
    def target_area(self) -> Polygon:
        """
        Get the target area as a shapely Polygon.

        Returns:
            Polygon: The target area polygon.
        """
        return self._target_area

    @target_area.setter
    def target_area(self, path: str) -> None:
        """
        Set the target area by loading a GeoJSON file.

        Args:
            path (str): Path to the GeoJSON file.

        Raises:
            FileNotFoundError: If the GeoJSON file does not exist.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"ERROR \t The geojson at {path} does not exist.")

        with open(path, 'r') as f:
            geojson_data = json.load(f)

        # Convert the GEOJSON data to a shapely object
        geometry = geojson_data['features'][0]['geometry']
        shapely_shape = shape(geometry)

        self._target_area = shapely_shape

    # Population density
    @property
    def population_density(self) -> str:
        """
        Get the path to the population density file.

        Returns:
            str: The file path for population density data.
        """
        return self._population_density

    @population_density.setter
    def population_density(self, path: str) -> None:
        """
        Set the population density file path.

        Args:
            path (str): Path to the population density file.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"ERROR \t The population density at {path} does not exist.")
        
        self._population_density = path

    # Destinations
    @property
    def destinations(self) -> list:
        """
        Get the destination points with weights.

        Returns:
            list: A list of tuples representing destination points (longitude, latitude).
        """
        return self._destinations

    @destinations.setter
    def destinations(self, path: str) -> None:
        """
        Set the destinations by loading a CSV file with destination points and weights.

        Args:
            path (str): Path to the CSV file with destinations.

        Raises:
            FileNotFoundError: If the CSV file does not exist.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"ERROR \t The CSV for destinations at {path} does not exist.")
        
        center_points = []
        
        with open(path, mode='r') as csv_file:
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

        self._destinations = center_points

    # Target Area Centroid Coordinates
    @property
    def centroid_coords(self) -> tuple:
        """
        Get the coordinates of the centroid of the target area.

        Returns:
            tuple: The (latitude, longitude) coordinates of the centroid.
        """
        centroid = self.target_area.centroid
        return centroid.y, centroid.x

    #######################################
    ######### Simulation setup ############
    #######################################

    def setup_simulation(self, taz_target_width_km: float, simulation_area_extension_km: float, population_to_ignore_share: float) -> None:
        """
        Set up the simulation by configuring the simulation bounding box (bbox), transportation analysis zone (TAZ) width, 
        and initializing the TAZ based on population.

        Args:
            taz_target_width_km (float): Target width of each TAZ in kilometers.
            simulation_area_extension_km (float): Extension in kilometers to apply to the simulation bounding box.
            population_to_ignore_share (float): The share of the population to ignore during the setup.
        """
        print(f"INFO \t SIMULATION SETUP")

        self.set_simulation_bbox(simulation_area_extension_km)
        self.set_taz_width(taz_target_width_km)
        self.init_taz(population_to_ignore_share)

        self.state = "initialized"

        print(f"INFO \t Simulation setup done. Make sure to rerun trip generation and distribution if needed.")        
        print(f" \t Simulation area - Width: {self.simulation_area_width} km | Height: {self.simulation_area_height} km | Pop: {self.traffic_zones['population'].sum()} | Destinations: {self.traffic_zones['destinations'].sum()}")
        print(f" \t TAZ - Number: {len(self.traffic_zones)} | Width: {self.taz_width} km | Height: {self.taz_height} km ")

    # Simulation bbox
    @property
    def simulation_bbox(self) -> tuple:
        """
        Get the current simulation bounding box (bbox).

        Returns:
            tuple: Coordinates of the bounding box as (minx, miny, maxx, maxy).
        """
        return self._simulation_bbox

    def set_simulation_bbox(self, simulation_area_extension_km: float) -> None:
        """
        Set the simulation bounding box (bbox) by extending the target area boundaries.

        Args:
            simulation_area_extension_km (float): Distance in kilometers by which to extend the simulation bbox.
        """
        print(f"INFO \t Extending the simulation bbox by {simulation_area_extension_km} km")

        # Get the bounding box of the shapefile defining the target area
        minx, miny, maxx, maxy = self.target_area.bounds

        def extend_bbox(minx: float, miny: float, maxx: float, maxy: float, km_extension: float) -> tuple:
            """
            Extend the bounding box (bbox) by a given extension in kilometers.

            Args:
                minx (float): Minimum x-coordinate (longitude).
                miny (float): Minimum y-coordinate (latitude).
                maxx (float): Maximum x-coordinate (longitude).
                maxy (float): Maximum y-coordinate (latitude).
                km_extension (float): Extension distance in kilometers.

            Returns:
                tuple: New extended bbox as (minx, miny, maxx, maxy).
            """
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

        self._simulation_bbox = new_minx, new_miny, new_maxx, new_maxy

    # Simulation Area: Width and Height
    @property
    def simulation_area_width(self) -> float:
        """
        Get the width of the simulation area based on the bounding box.

        Returns:
            float: The width of the simulation area in kilometers.
        """
        minx, miny, maxx, maxy = self.simulation_bbox
        return geodesic((minx, miny), (maxx, miny)).kilometers

    @property
    def simulation_area_height(self) -> float:
        """
        Get the height of the simulation area based on the bounding box.

        Returns:
            float: The height of the simulation area in kilometers.
        """
        minx, miny, maxx, maxy = self.simulation_bbox
        return geodesic((minx, miny), (minx, maxy)).kilometers

    # TAZ: Width, Height, Number of zones
    @property
    def taz_width(self) -> float:
        """
        Get the current width of each transportation analysis zone (TAZ).

        Returns:
            float: The width of the TAZ in kilometers.
        """
        return self._taz_width

    def set_taz_width(self, taz_target_width_km: float) -> None:
        """
        Set the width of the transportation analysis zones (TAZ) based on the target width.

        Args:
            taz_target_width_km (float): Target width of each TAZ in kilometers.
        """
        # Compute the number of integer segments close to the target width
        n = round(self.simulation_area_width / taz_target_width_km)

        # Calculate the actual segment length
        l = self.simulation_area_width / n

        self._taz_width = l

    @property
    def taz_height(self) -> float:
        """
        Get the height of each transportation analysis zone (TAZ).

        Returns:
            float: The height of the TAZ in kilometers.
        """
        return self.simulation_area_height / self.n_subdivisions

    @property
    def n_subdivisions(self) -> int:
        """
        Get the number of subdivisions (TAZ zones) along the height of the simulation area.

        Returns:
            int: Number of subdivisions.
        """
        return int(self.simulation_area_width / self.taz_width)

    # TAZ Initialization
    @property
    def traffic_zones(self) -> pd.DataFrame:
        """
        Get the DataFrame representing the traffic zones (TAZ).

        Returns:
            pd.DataFrame: A DataFrame containing traffic zone data.
        """
        return self._traffic_zones

    def init_taz(self, population_to_ignore_share: float = 0.0) -> None:
        """
        Initializes the Traffic Analysis Zones (TAZs) and sets up associated features.

        Args:
            population_to_ignore_share (float): Fraction of total population to ignore 
                (between 0 and 1). TAZs with cumulative population below this share will 
                be removed. Default is 0.0.
        
        Returns:
            None

        Prints:
            Information logs during the setup process.
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
                    out_image, out_transform = rasterio.mask.mask(src, [bbox_gdf.geometry.values[0]], crop=True)
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
                shapefile_geometry = self.target_area
                center_point = Point((center_lat, center_lon))

                # Check if the point is within the MultiPolygon
                is_within = shapefile_geometry.contains(center_point)

                # 6. Append everything
                grid_data.append({
                    'id': zone_id, 
                    'geometric_center': (center_lat, center_lon), 
                    'bbox': bbox_geom, 
                    'population': total_population, 
                    'destinations': n_destinations, 
                    'is_within_target_area': is_within
                })

        df = pd.DataFrame(grid_data)

        # Delete the sparsely populated TAZs, such that the sum of the less populated is below a threshold
        if population_to_ignore_share > 0.0 and population_to_ignore_share < 1.0:
            print(f"INFO \t Deleting sparsely populated TAZs")

            # Calculate the total population
            total_population = df['population'].sum()

            # Calculate the population limit based on the percentage
            population_limit = population_to_ignore_share * total_population

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

        # Assign the DataFrame to the traffic zones
        self._traffic_zones = pd.DataFrame(df)

    #######################################
    ########## Trip Generation ############
    #######################################

    def trip_generation(self, n_trips_per_inhabitant: float) -> None:
        """
        Generates the number of trips for each traffic zone based on the population and trips per inhabitant.

        Args:
            n_trips_per_inhabitant (float): Number of trips per inhabitant in each traffic zone.
                Must be greater than 0.

        Raises:
            RuntimeError: If the trip generation is not performed after simulation setup.
            ValueError: If the number of trips per inhabitant is not greater than 0.

        Returns:
            None

        Prints:
            Information logs during and after trip generation, including the total number of trips.
        """
        print(f"INFO \t TRIP GENERATION")

        # Check if the simulation is in the correct state
        if self.state != "initialized":
            raise RuntimeError(f"ERROR \t Trip generation must be performed right after simulation setup")

        # Check for valid trips per inhabitant
        if n_trips_per_inhabitant <= 0.0:
            raise ValueError(f"ERROR \t Trips per inhabitant must be greater than 0")

        # Load the traffic_zones dataframe
        df = self.traffic_zones

        # Calculate the number of trips and append them to the dataframe
        df['n_outflows'] = df['population'].apply(lambda x: int(x * n_trips_per_inhabitant))

        # Update the traffic zones with the new trip data
        self._traffic_zones = df

        # Update the state
        self.state = "generation_done"

        # Print summary information
        print(f"INFO \t Trip generation done. Make sure to rerun trip distribution if needed.")
        print(f"\t Total number of trips: {df['n_outflows'].sum()}")

    #######################################
    ######## Trip Distribution ############
    #######################################

    def trip_distribution(self, model: str, ors_key: str = None, attraction_feature: str = "population", cost_feature: str = "distance_road", batch_size: int = 49, km_per_capita_offset: float = 0, road_to_euclidian_ratio: float = 1.63) -> None:
        """
        Distributes trips between Traffic Analysis Zones (TAZ) based on a spatial interaction model.

        This function performs road distance calculations using either the OpenRouteService (ORS) or an empirical road-to-Euclidean 
        ratio, and computes flows between zones. The function supports multiple spatial interaction models (gravity models, radiation 
        model, and radius-based models), allowing the user to choose an appropriate one based on the desired behavior of the trips.

        Args:
            model (str): The type of spatial interaction model to apply for trip distribution. Available models are:
                - 'gravity_power_1'
                - 'gravity_exp_1'
                - 'gravity_exp_01'
                - 'gravity_exp_scaled'
                - 'radiation'

            ors_key (str): OpenRouteService API key for obtaining road distances. If None, road distance will be estimated using the road-to-Euclidean ratio. Default is None.

            attraction_feature (str): Feature used for the attraction calculation in the spatial interaction model. Options are 'population' or 'destinations'. Default is 'population'.

            cost_feature (str): Feature used for the cost calculation in the spatial interaction model. Default is 'distance_road'. Options are:
                - 'distance_road': Road distance between TAZ
                - 'time_road': Travel time by road
                - 'distance_centroid': Euclidean distance between TAZ centroids

            batch_size (int): The number of origins/destinations processed per ORS request if `ors_key` is provided. Default is 49.

            km_per_capita_offset (float): Offset added to the per capita kilometers in inflows/outflows calculation. Default is 0.

            road_to_euclidean_ratio (float): Empirical ratio to estimate road distance based on Euclidean distance if `ors_key` is not used. Default is 1.63.

        Raises:
            RuntimeError: If trip generation has not been completed before distribution.
            ValueError: If an unknown attraction or cost feature is specified, or if an unknown spatial interaction model is specified. 
        """

        print(f"INFO \t TRIP DISTRIBUTION")

        if self.state != "generation_done":
            raise RuntimeError(f"ERROR \t Trip distribution must be performed right after trip generation")

        ############ Get TAZ data ############

        df = self.traffic_zones

        # Check if trip generation has been performed
        if not 'n_outflows' in df.columns:
            print(f"ERROR \t Traffic Analysis Zones do not contain the number of trips: trip generation must be performed before trip distribution.")
            return

        ############ Get Road Distance ############

        print(f"INFO \t Getting road distance")

        # Extract coordinates of the centroid
        coordinates = [list(coord) for coord in df['geometric_center']] 
        num_coordinates = len(coordinates)

        # Initialize the full matrices
        durations = np.zeros((num_coordinates, num_coordinates))
        distances = np.zeros((num_coordinates, num_coordinates))

        # If the user provides an ORS key
        if not ors_key == None:
            print(f"INFO \t Calculating distance by road using ORS matrix request")

            # Check if the number of coordinates exceeds the batch size
            if num_coordinates > batch_size:
                print(f"ALERT \t {num_coordinates} origins/destinations: this number is greater than the batch size set at {batch_size}. Multiple ORS requests are needed.")

            # Initialize ORS client
            client = openrouteservice.Client(key=ors_key)  # Replace with your ORS API key

            # Split the coordinates into manageable batches
            coordinate_batches = [coordinates[i:i+batch_size] for i in range(0, len(coordinates), batch_size)]

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
        else:
            print(f"INFO \t Distance by road calculated using empirical road-to-euclidian ratio equal to {road_to_euclidian_ratio}")

        # Define constants for distance and time within two points located in the same zone
        minx, miny, maxx, maxy = self.simulation_bbox

        # Prepare lists to hold the data
        origin_ids = []
        destination_ids = []
        flows = []
        travel_times = []
        travel_distances = []
        travel_distances_euclidian = []

        # Populate the data based on the matrix response or on the road/euclidian distance ratio
        ors_errors = 0
        for i, origin_id in enumerate(df['id']):
            for j, destination_id in enumerate(df['id']):
                origin_ids.append(origin_id)
                destination_ids.append(destination_id)

                flows.append(0.0)  # Placeholded for the flow for all origin-destination pairs                

                # Calculate euclidian travel distance and append data
                point1 = df.loc[df['id'] == origin_id, 'geometric_center'].iloc[0]
                point2 = df.loc[df['id'] == destination_id, 'geometric_center'].iloc[0]

                euclidian_distance = geodesic(point1, point2).kilometers

                travel_distances_euclidian.append(euclidian_distance)

                # For all zero distances (the case for all combinations if ORS calculation has not been performed)
                if (math.isnan(distances[i][j]) or distances[i][j] == .0) and (origin_id != destination_id):
                    distance = euclidian_distance * road_to_euclidian_ratio

                    travel_distances.append(euclidian_distance * road_to_euclidian_ratio)
                    travel_times.append(distance / 30 * 60)

                    # If ors calculation was done, calculate the number of unresolved locations
                    if ors_key != None:
                        ors_errors = ors_errors + 1
                        
                else:
                    travel_distances.append(distances[i][j] / 1000)  # Convert meters to kilometer
                    travel_times.append(durations[i][j] / 60)  # Convert seconds to minutes

        if ors_errors:
            print(f"ALERT \t ORS was unable to calculate distance or resolve {ors_errors} routes. Using road to euclidian distance ratio instead and a travel speed of 30 km/h. This could affect the model reliability!")

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
        unit_surface_alert = False # Flag in the case of the auto-calibrated gravity model

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
                raise ValueError(f"ERROR \t Attraction feature is unknown.")

            # SIM input: cost
            if cost_feature == 'distance_road':
                cost_list = origin_rows['Travel Distance (km)'].tolist() # Extract the "Travel Distance (km)" column into a list
            elif cost_feature == 'time_road':
                cost_list = origin_rows['Travel Time (min)'].tolist() # Extract the "Travel Time (min)" column into a list
            elif cost_feature == 'distance_centroid':
                cost_list = origin_rows['Centroid Distance (km)'].tolist() # Extract the "Centroid Distance (km)" column into a list
            else:
                raise ValueError(f"ERROR \t Cost feature is unknown.")  

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
            elif model == 'gravity_exp_scaled':
                # Gravity model auto-calibrated 
                # https://doi.org/10.1371/journal.pone.0045985
                # https://doi.org/10.1016/j.jtrangeo.2015.12.008
                unit_surface_km2 = self.taz_width*self.taz_height
                if unit_surface_km2 < 5 and not unit_surface_alert:
                    print("ALERT \t The average unit surface area is less than 5 km2, which may cause the scaling law of the gravity model to be invalid.")
                    unit_surface_alert = True
                flows = hlp.prod_constrained_gravity_exp(
                    origin_n_trips = n_outflows,
                    dest_attractivity_list = dest_att_list,                
                    cost_list = cost_list, 
                    beta = 0.3 * unit_surface_km2**(-0.18) )
            elif model == 'radiation':
                flows = hlp.prod_constrained_radiation(
                    origin_n_trips = n_outflows,
                    origin_attractivity = att_origin,
                    dest_attractivity_list = dest_att_list,                
                    cost_list = cost_list)
            else:
                raise ValueError(f"ERROR \t Spatial interaction model '{model}' is unknown.")

            # Update the flows column where row_id equals 1
            flows_df.loc[flows_df['Origin'] == origin_id, 'Flow'] = flows[:len(flows_df.loc[flows_df['Origin'] == origin_id])]

        ############ Append flow data ############

        self._flows = flows_df

        ############ Append Aggregated data to TAZ ############

        n_outflows = []
        n_inflows = [] 
        pkm_outflows = []
        pkm_inflows = []
        km_per_capita_outflows = []
        km_per_capita_inflows = []

        # Iterate over the TAZ and append data
        for index, row in df.iterrows():

            # Append values related to the origin (outflows)            
            out_df = flows_df[flows_df['Origin'] == row['id']].copy()
            out_df['Distance_Flow_Product'] = (out_df['Travel Distance (km)'] + km_per_capita_offset) * out_df['Flow']

            outflow_sum = out_df['Flow'].sum()
            distance_flow_product_sum_out = out_df['Distance_Flow_Product'].sum()

            n_outflows.append(outflow_sum)
            pkm_outflows.append(distance_flow_product_sum_out)
            if outflow_sum != 0:
                km_per_capita_outflows.append(distance_flow_product_sum_out / outflow_sum)
            else:
                km_per_capita_outflows.append(0)

            # Append values related to the destination (inflows)            
            in_df = flows_df[flows_df['Destination'] == row['id']].copy()
            in_df['Distance_Flow_Product'] = (in_df['Travel Distance (km)'] + km_per_capita_offset) * in_df['Flow']

            inflow_sum = in_df['Flow'].sum()
            distance_flow_product_sum_in = in_df['Distance_Flow_Product'].sum()

            n_inflows.append(inflow_sum)
            pkm_inflows.append(distance_flow_product_sum_in)
            if inflow_sum != 0:
                km_per_capita_inflows.append(distance_flow_product_sum_in / inflow_sum)
            else:
                km_per_capita_inflows.append(0)

        # Add a new column with values from the list
        self._traffic_zones['n_outflows'] = n_outflows
        self._traffic_zones['n_inflows'] = n_inflows
        self._traffic_zones['pkm_outflows'] = pkm_outflows
        self._traffic_zones['pkm_inflows'] = pkm_inflows
        self._traffic_zones['km_per_capita_outflows'] = km_per_capita_outflows
        self._traffic_zones['km_per_capita_inflows'] = km_per_capita_inflows

        self.state = "distribution_done"

        print(f"INFO \t Trip distribution done.")
        print(f"\t Passenger-km (road-based): {self.pkm} km | Av. distance travelled (road-based ): {self.km_per_capita} km")
        print(f"\t Passenger-km (centroid-based): {self.pkm_centroid} km | Av. distance travelled (centroid-based): {self.km_per_capita_centroid} km")

    @property
    def flows(self):
        """
        Get the flows DataFrame.

        Returns:
            pd.DataFrame: The flows data containing various metrics.
        """
        return self._flows

    @property
    def pkm(self):
        """
        Calculate the total person-kilometers-moved (PKM) from traffic zones.

        Returns:
            float: The total PKM, which is the sum of outflows from traffic zones.
        """
        return self.traffic_zones['pkm_outflows'].sum()

    @property
    def pkm_centroid(self):
        """
        Calculate the weighted centroid distance for PKM.

        Returns:
            float: The weighted average distance (in kilometers) of flows from the centroid.
        """
        return self.flows['Centroid Distance (km)'].dot(self.flows['Flow'])  

    @property
    def pkm_error(self):
        """
        Estimate the error in PKM calculation based on flow and TAZ dimensions.

        Returns:
            float: The estimated error in PKM, calculated using error propagation based on TAZ width and height.
        """
        flows = self.flows['Flow']
        flows_squared = flows ** 2

        return 2 * np.sqrt(self.taz_width**2 + self.taz_height**2) * np.sqrt(flows_squared.sum()) 

    @property
    def km_per_capita_error(self):
        """
        Calculate the error in kilometers per capita based on PKM error.

        Returns:
            float: The estimated error in kilometers per capita.
        """
        return self.km_per_capita * (self.pkm_error / self.pkm)

    @property
    def km_per_capita(self):
        """
        Calculate kilometers per capita based on total PKM and outflows.

        Returns:
            float: The average kilometers per capita, calculated as PKM divided by total outflows.
        """
        return self.pkm / self.traffic_zones['n_outflows'].sum()

    @property
    def km_per_capita_centroid(self):
        """
        Calculate the weighted average kilometers per capita based on centroid distances.

        Returns:
            float: The average kilometers per capita weighted by flow from the centroid distance.
        """
        return np.average(self.flows['Centroid Distance (km)'], weights=self.flows['Flow'])

    #######################################
    ############## Routing ################
    #######################################

    def allocate_routes(self, ors_key: str) -> None:
        """
        Allocate ORS routes to origin-destination pairs based on flow data.

        This function computes the routes between specified origins and destinations
        using the OpenRouteService API. The computed routes are stored in the 'Geometry'
        column of the flows DataFrame. The function caches previously computed routes to
        improve efficiency.

        Args:
            ors_key (str): The API key for accessing OpenRouteService.

        Raises:
            RuntimeError: If the trip distribution has not been completed before routing.

        Returns:
            None: This function modifies the flows DataFrame in place by adding a 
                  'Geometry' column with the route geometries.
        """
        print(f"INFO \t Allocation of ORS routes to origin-destination pairs (routing)")

        if self.state != "distribution_done":
            raise RuntimeError(f"ERROR \t Routing cannot be performed before trip distribution")

        flows = self.flows
        taz = self.traffic_zones

        # Add a new column for the route geometry
        flows['Geometry'] = None

        # Initialize ORS client
        client = openrouteservice.Client(key=ors_key)  # Replace with your ORS API key

        # Create a dictionary to store previously calculated routes (avoids recalculating when origin and destination are swapped)
        route_cache = {}

        i = 0
        for index, row in flows.iterrows():
            i += 1

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


    #######################################
    ### Post-processing & visualisation ###
    #######################################

    def km_per_capita_histogram(self, bin_width_km: float) -> pd.DataFrame:
        """
        Generates a histogram of centroid and travel distances per capita.

        Args:
            bin_width_km (float): The width of the bins in kilometers.

        Returns:
            pd.DataFrame: A DataFrame containing the distance bins, centroid distance counts, 
                           and travel distance counts.
        """
        centroid_distance = self.flows['Centroid Distance (km)']
        travel_distance = self.flows['Travel Distance (km)']
        weights = self.flows['Flow']

        bin_edges = np.arange(0, travel_distance.max()+1, bin_width_km)  # Bins of width 10, from 0 to 10

        # Calculate the histogram
        centroid_distance_counts, bin_edges = np.histogram(centroid_distance, bins=bin_edges, weights=weights)
        travel_distance_counts, bin_edges = np.histogram(travel_distance, bins=bin_edges, weights=weights)

        # Calculate the bin centers (optional)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Create a DataFrame to store the bin centers and counts
        hist_df = pd.DataFrame({
            'Distance (km)': bin_centers,
            'Centroid ': centroid_distance_counts,
            'Travel (road)': travel_distance_counts
        })

        return hist_df

    def setup_to_map(self) -> folium.Map:
        """
        Generates a folium map with simulation setup properties, including administrative boundaries, 
        simulation bounding box, and feature groups for destinations and population.

        Returns:
            folium.Map: A folium map object with the specified properties.
        """
        print(f"INFO \t Generating folium map with simulation setup properties")

        df = self.traffic_zones

        # 1. Create an empty map
        m1 = folium.Map(location=self.centroid_coords, zoom_start=12, tiles='CartoDB Positron', control_scale=True)

        # 2. Add Administrative Boundaries
        def style_function(feature):
            return {
                'color': 'blue',
                'weight': 3,
                'fillColor': 'none',
            }
        
        folium.GeoJson(self.target_area, name='Administrative boundary', style_function=style_function).add_to(m1)

        # 3. Add Simulation bbox
        minx, miny, maxx, maxy = self.simulation_bbox
        folium.Rectangle(bounds=[[miny, minx], [maxy, maxx]], fill=True, fill_opacity=0, color='blue', weight=2).add_to(m1)

        # 5. Add rectangles (using apply)
        def add_rectangle(row, colormap, col, map_obj):
            bbox_polygon = row['bbox']
            bbox_coords = bbox_polygon.bounds
            folium.Rectangle(
                bounds=[(bbox_coords[1], bbox_coords[0]), (bbox_coords[3], bbox_coords[2])],
                color=None, fill=True, fill_color=colormap(row[col]), fill_opacity=0.7
            ).add_to(map_obj)

        # Normalize data for destinations and population
        linear1 = cm.LinearColormap(["white", "yellow", "red"], vmin=df['destinations'].min(), vmax=df['destinations'].max())
        linear2 = cm.LinearColormap(["white", "yellow", "red"], vmin=df['population'].min(), vmax=df['population'].max())

        # Create FeatureGroups for destinations and population
        destinations_group = folium.FeatureGroup(name='Number of Destinations', show=False)
        population_group = folium.FeatureGroup(name='Number of People', show=False)

        # Add destinations rectangles to the group
        df.apply(lambda row: add_rectangle(row, linear1, 'destinations', destinations_group), axis=1)

        # Add population rectangles to the group
        df.apply(lambda row: add_rectangle(row, linear2, 'population', population_group), axis=1)

        # Add the FeatureGroups to the map
        destinations_group.add_to(m1)
        population_group.add_to(m1)

        # Add color scales
        linear1.caption = 'Number of destinations'
        linear1.add_to(m1)
        
        linear2.caption = 'Number of people'
        linear2.add_to(m1)

        # 6. Add Layer Control
        folium.LayerControl().add_to(m1)
        
        return m1

    def trip_generation_to_map(self) -> folium.Map:
        """
        Generates a folium map from trip generation results, including administrative boundaries 
        and TAZ boundaries, and visualizing the number of outflows.

        Returns:
            folium.Map: A folium map object showing trip generation results.
        """
        print(f"INFO \t Generating folium map from trip generation results")

        df = self.traffic_zones

        # 1. Create an empty map
        m2 = folium.Map(location=self.centroid_coords, zoom_start=12, tiles='CartoDB Positron', control_scale=True) # Create the map

        # 2. Add Administrative Boundaries
        def style_function(feature):
            return {
                'color': 'blue',
                'weight': 3,
                'fillColor': 'none',
            }
        
        folium.GeoJson(self.target_area, name='Administrative boundary', style_function=style_function).add_to(m2)

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
        self.traffic_zones.apply(add_rectangle, axis=1)

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

        return m2

    def trip_distribution_to_map(self, trip_id: int) -> folium.Map:
        """
        Generates a folium map using trip distribution for a specified trip ID, visualizing the number 
        of trips from the given trip ID.

        Args:
            trip_id (int): The ID of the trip to visualize.

        Returns:
            folium.Map: A folium map object showing trip distribution for the specified trip ID.
        """
        print(f"INFO \t Generating folium map using trip distribution for trip id {trip_id}")        

        m3 = folium.Map(location=self.centroid_coords, zoom_start=12, tiles='CartoDB Positron', control_scale=True) # Create the map

        # Add administrative boundaries

        # Define style function to only show lines
        def style_function(feature):
            return {
                'color': 'blue',  # Set line color
                'weight': 3,      # Set line weight
                'fillColor': 'none',  # Set fill color to 'none'
            }

        folium.GeoJson(self.target_area, name='Administrative boundary', style_function=style_function).add_to(m3)

        # Add TAZ boundaries

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
            ).add_to(m3)

        # Apply the function to each row in the DataFrame
        self.traffic_zones.apply(add_rectangle, axis=1)

        #Add flow

        df = self.flows
        group = df[df['Origin'] == trip_id]

        if len(group) == 0:
            print("No TAZ")

        feature_group = folium.FeatureGroup(name=f'Number of trips from {trip_id}')
    
        # Add flows to the feature group
        for idx, row in group.iterrows():
            linear = cm.LinearColormap(["white", "yellow", "red"], vmin=group['Flow'].min(), vmax=group['Flow'].max())

            bbox_polygon = self.traffic_zones[self.traffic_zones['id'] == row['Destination']]['bbox'].values[0]
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

        feature_group.add_to(m3)
        folium.LayerControl().add_to(m3)

        return m3
        