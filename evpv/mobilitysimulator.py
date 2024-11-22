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

from evpv.vehicle import Vehicle
from evpv.vehiclefleet import VehicleFleet
from evpv.region import Region
from evpv import helpers as hlp

class MobilitySimulator:
    """
    A class to perform a mobility demand simulation for a given region.

    This class simulates the demand of commuting demand based on a given vehicle fleet and region, supporting various methods for 
    allocating vehicles spatially and performing trip distribution using spatial interaction modelling.

    Key Features:
    - Vehicle allocation: Allocates vehicles across zones of a region based on user-defined methods, supporting the addition of randomness.
    - Trip distribution: Supports various spatial interaction models for trip distribution, such as the gravity and radiation models, allowing for customization of attraction and cost features to generate realistic commuting demand.
    - Self-calibrated gravity model: Integrates a self-calibrated gravity model that can be used without calibration, following the work of Lenormand et. al. 2016  
    - Routing and time/distance calculation: Enables flexible distance computation through both euclidean and road-based distances, with optional integration of OpenRouteService (ORS) API for advanced routing.
    - Distance offset: Allows for adding a distance offset added to the distance calculated for commuting.

    Note: This class assumes the region is pre-defined and vehicle allocation parameters are specified in a dictionary. The use of the `ors_key` parameter enables routing with real road network data, while omitting it defaults to basic distance calculations.
    """
    
    def __init__(self, vehicle_fleet: VehicleFleet, region: Region, vehicle_allocation_params: dict, trip_distribution_params: dict):
        """
        Initializes the MobilitySimulator class.
        
        Args:
            vehicle_fleet (VehicleFleet): An instance of the VehicleFleet class.
            region (Region): An instance of the Region class.
            vehicle_allocation_params (dict): Method for vehicle spatial allocation generation. Keys:
                'method' (str): The method used for trip generation (e.g., 'population').
                'randomness' (float): A value between 0 and 1 to introduce randomness.
            trip_distribution_params (dict): Parameters specific to trip distribution. Keys:
                'road_to_euclidian_ratio' (float): Ratio of road distance to Euclidean distance.
                'ors_key' (str or None): Optional ORS (OpenRouteService) API key for routing.
                'model_type' (str): Type of spatial interaction model ('gravity' or 'radiation').
                'attraction_feature' (str): Feature to attract trips (e.g., 'workplaces').
                'cost_feature' (str): Feature representing the cost of travel (e.g., 'distance_road').
                'distance_offset_km' (float): Distance added to the distance calculated through trip distribution.
        """
        print("=========================================")
        print(f"INFO \t Creation of a MobilitySimulator object.")
        print("=========================================")

        self.vehicle_fleet = vehicle_fleet
        self.region = region
        self.vehicle_allocation_params = vehicle_allocation_params        
        self.trip_distribution_params = trip_distribution_params

        print(f"INFO \t Successful initialization of input parameters.")

        self._flows = None # To store flows between traffic zones
        self._aggregated_zone_metrics = None # To store aggregated metrics for each zone (like vehicle-km or average distance travelled)

    # Results properties (read-only)

    @property
    def flows(self) -> pd.DataFrame:
        """pd.DataFrame: The flows of the trip distribution model."""
        return self._flows

    @property
    def aggregated_zone_metrics(self) -> pd.DataFrame:
        """pd.DataFrame: The aggregated_zone_metrics of the trip distribution model."""
        return self._aggregated_zone_metrics

    # Properties and Setters 

    @property
    def vehicle_fleet(self) -> VehicleFleet:
        """VehicleFleet: The vehicle fleet used in the simulation."""
        return self._vehicle_fleet

    @vehicle_fleet.setter
    def vehicle_fleet(self, value: VehicleFleet):
        if not isinstance(value, VehicleFleet):
            raise ValueError("vehicle_fleet must be an instance of the VehicleFleet class.")
        self._vehicle_fleet = value

    @property
    def region(self) -> Region:
        """Region: The region used in the simulation."""
        return self._region

    @region.setter
    def region(self, value: Region):
        if not isinstance(value, Region):
            raise ValueError("region must be an instance of the Region class.")
        self._region = value

    @property
    def vehicle_allocation_params(self) -> str:
        """str: The method used for trip generation (e.g., 'population')."""
        return self._vehicle_allocation_params

    @vehicle_allocation_params.setter
    def vehicle_allocation_params(self, value: str):
        # Check if `value` is a dictionary
        if not isinstance(value, dict):
            raise ValueError("Trip generation method must be a dictionary.")

        # Validate required keys
        if 'method' not in value:
            raise ValueError("Missing required trip generation parameter: 'method'.")
        if 'randomness' not in value:
            raise ValueError("Missing required trip generation parameter: 'randomness'.")

        # Validate types
        if not isinstance(value['method'], str) or not value['method']:
            raise ValueError("Trip generation method must be a non-empty string.")
        
        if not isinstance(value['randomness'], (float, int)) or not (0 <= value['randomness'] <= 1):
            raise ValueError("Randomness must be a float between 0 and 1.")

        # If all checks pass, set the parameter
        self._vehicle_allocation_params = value

    @property
    def trip_distribution_params(self) -> dict:
        """dict: Parameters used for trip distribution."""
        return self._trip_distribution_params

    @trip_distribution_params.setter
    def trip_distribution_params(self, value: dict):
        # Define required keys and expected types for each key
        required_keys = {
            'road_to_euclidian_ratio': (float, int),  # ratio should be numeric
            'ors_key': (str, type(None)),             # optional API key, string or None
            'model_type': str,                        # model type as a string
            'attraction_feature': str,                # feature name as a string
            'cost_feature': str,                      # feature name as a string
            'distance_offset_km': (float)
        }

        # Check if `value` is a dictionary
        if not isinstance(value, dict):
            raise ValueError("Trip distribution parameters must be a dictionary.")

        # Validate each required key is present with the correct type
        for key, expected_type in required_keys.items():
            if key not in value:
                raise ValueError(f"Missing required trip distribution parameter: '{key}'.")
            if not isinstance(value[key], expected_type):
                raise TypeError(f"Parameter '{key}' must be of type {expected_type}.")

        # Additional checks for specific parameter values
        if value['road_to_euclidian_ratio'] <= 0:
            raise ValueError("road_to_euclidian_ratio must be a positive number.")
        # Check the connection with ORS
        if value['ors_key'] is not None:
            if not self.is_valid_ors_connection(value['ors_key']):
                raise ValueError("Impossible to connect to ORS. Check the online status of the service (https://status.openrouteservice.org/) and the validity of your ORS key.")

        # If all checks pass, set the parameter
        self._trip_distribution_params = value

    # Helpers

    def is_valid_ors_connection(self, ors_key: str) -> bool:
        """Check if the ORS key is valid by creating a client.

        Args:
            ors_key (str): The ORS API key to validate.

        Returns:
            bool: True if the key is valid, False otherwise.
        """
        try:
            # Attempt to create an ORS client with the provided key
            client = openrouteservice.Client(key=ors_key)
            # Perform a simple request to check validity
            routes = client.directions(((8.34234,48.23424),(8.34423,48.26424)))
            return True  # If no exceptions, the key is valid
        except Exception as e:
            print(f"Error while trying to establish a connection with ORS: {e}")
            return False

    # Vehicle allocation methods

    def vehicle_allocation(self) -> pd.DataFrame:
        """Allocates vehicle (number of outflows) to traffic zones.
        
        This method returns a new DataFrame containing the `id` of each zone and 
        the allocated number of outflows (`n_outflows`).

        Returns:
            pd.DataFrame: A DataFrame with zone metrics, including `id` and `n_outflows`.
        """
        print(f"INFO \t Allocating vehicles to traffic zones...")

        method = self.vehicle_allocation_params['method']
        randomness = self.vehicle_allocation_params['randomness']
        zones = self.region.traffic_zones.copy()  # Make a copy to avoid direct modification
        n_outflows = self.vehicle_fleet.total_vehicles

        method_vehicle_count = int(n_outflows * (1 - randomness))
        random_vehicle_count = n_outflows - method_vehicle_count

        print(f"INFO \t Method-based allocation of {method_vehicle_count} vehicles...")
        
        # Initial vehicle allocation based on specified method
        if method == 'population':
            total_population = zones['n_people'].sum()
            zones['n_outflows'] = zones['n_people'].apply(
                lambda x: round(method_vehicle_count * (x / total_population)) if total_population > 0 else 0
            )
        else:
            print(f"ERROR \t The vehicle allocation method {method} is unknown.")

        # Random allocation for remaining vehicles
        if randomness != 0:
            print(f"INFO \t Random allocation of {random_vehicle_count} vehicles...")
            for _ in range(random_vehicle_count):
                selected_zone = zones.sample().index[0]
                zones.at[selected_zone, 'n_outflows'] += 1

        # Correct the allocation to match total outflow count
        difference = n_outflows - zones['n_outflows'].sum()
        if difference != 0:
            print(f"ALERT \t Adjusting difference of {difference} vehicles due to rounding...")
            zones = self._adjust_vehicle_allocation(difference, zones)

        # Update the traffic zones with the new data
        self._aggregated_zone_metrics = zones[['id', 'n_outflows']]

        # Print 
        print(f"\t > Total vehicles: {zones['n_outflows'].sum()}")
        print(f"\t > Vehicles per zone: {zones['n_outflows'].mean()} ± {zones['n_outflows'].std()}")

        # ASCII Histogram of vehicle distribution
        # Should be a close to a normal distribution when randomness is 100%
        bins = np.histogram_bin_edges(zones['n_outflows'], bins='auto')
        hist, bin_edges = np.histogram(zones['n_outflows'], bins=bins)
        
        print("\t > Distribution of zones per vehicle count:") 
        for count, edge in zip(hist, bin_edges[:-1]):
            bar = "#" * (count * 2)  # Scale the bar for better visibility
            print(f"\t {int(edge):>4} - {int(edge + (bin_edges[1] - bin_edges[0])):<4}: {bar}")

    def _adjust_vehicle_allocation(self, difference: int, zones: pd.DataFrame) -> pd.DataFrame:
        """Adjusts vehicle allocation to match the total count.

        Args:
            difference (int): The difference between allocated and target outflow counts.
            zones (pd.DataFrame): DataFrame with current outflows per zone.

        Returns:
            pd.DataFrame: Adjusted zones DataFrame with corrected outflows.
        """
        if difference > 0:
            # Add outflows based on current distribution
            for i in range(difference):
                total_current_outflows = zones['n_outflows'].sum()
                probabilities = zones['n_outflows'] / total_current_outflows
                selected_zone = zones.sample(weights=probabilities).index[0]
                zones.at[selected_zone, 'n_outflows'] += 1
        elif difference < 0:
            # Remove outflows based on current distribution
            for i in range(-difference):
                total_current_outflows = zones['n_outflows'].sum()
                probabilities = zones['n_outflows'] / total_current_outflows
                # Prevent negative number of vehicles
                while True:
                    selected_zone = zones.sample(weights=probabilities).index[0]
                    if zones.at[selected_zone, 'n_outflows'] > 0:
                        zones.at[selected_zone, 'n_outflows'] -= 1
                        break
        return zones

    # Trip distribution methods

    def trip_distribution(self, batch_size: int = 49) -> None:
        """
        Distributes trips between traffic zones based on a spatial interaction model. 
        """
        print(f"INFO \t Trip distribution...")

        # Get relevant parameters

        model = self.trip_distribution_params['model_type']
        ors_key = self.trip_distribution_params['ors_key']
        road_to_euclidian_ratio = self.trip_distribution_params['road_to_euclidian_ratio']
        attraction_feature = self.trip_distribution_params['attraction_feature']
        cost_feature = self.trip_distribution_params['cost_feature']
        vkt_offset = self.trip_distribution_params['distance_offset_km']        
        
        if self._aggregated_zone_metrics is None: # Check if the Region contains the number of vehicles
            raise RuntimeError(f"ERROR \t Please perfom vehicle allocation before trip distribution.")

        # Create a dataframe with region attributes and vehicle outflows for convenience

        df = self.region.traffic_zones.copy()        
        df['n_outflows'] = self._aggregated_zone_metrics['n_outflows'] # Add the new column 'n_outflows' to the copied dataframe

        # Calculate the distances and travel times between traffic zones using ORS or road_to_euclidian_ratio

        print(f"INFO \t Calculating distance/time between traffic zones...")

        flows_df = self._get_travel_data(df, ors_key, road_to_euclidian_ratio, batch_size)

        # Apply the spatial interaction model

        print(f"INFO \t Applying the '{model}' spatial interaction model...")

        for origin in flows_df['Origin'].unique():
            # Filter rows for the current origin
            origin_rows = flows_df[flows_df['Origin'] == origin]
            origin_id = origin_rows.iloc[0]['Origin']

            # SIM input: outgoing trips
            n_outflows = df.loc[df['id'] == origin_id, 'n_outflows'].values[0]

            # Get attraction and cost features
            att_origin, dest_att_list, cost_list = self._get_attraction_cost_data(df, origin_id, origin_rows, attraction_feature, cost_feature)

            # Calculate the flows based on the model
            flows = self._apply_spatial_interaction_model(model, n_outflows, att_origin, dest_att_list, cost_list)
      
            # Update the flows column where row_id equals 1
            flows_df.loc[flows_df['Origin'] == origin_id, 'Flow'] = flows[:len(flows_df.loc[flows_df['Origin'] == origin_id])]

        # Append flow data 
        self._flows = flows_df

        # Add aggregated data to the region object

        n_outflows = []
        n_inflows = [] 
        fkt_outflows = []
        fkt_inflows = []
        vkt_outflows = []
        vkt_inflows = []

        # Iterate over the TAZ and append data
        for index, row in df.iterrows():

            # Append values related to the origin (outflows)            
            out_df = flows_df[flows_df['Origin'] == row['id']].copy()
            out_df['Distance_Flow_Product'] = (out_df['Travel Distance (km)'] + vkt_offset) * out_df['Flow']

            outflow_sum = out_df['Flow'].sum()
            distance_flow_product_sum_out = out_df['Distance_Flow_Product'].sum()

            n_outflows.append(outflow_sum)
            fkt_outflows.append(distance_flow_product_sum_out)
            if outflow_sum != 0:
                vkt_outflows.append(distance_flow_product_sum_out / outflow_sum)
            else:
                vkt_outflows.append(0)

            # Append values related to the destination (inflows)            
            in_df = flows_df[flows_df['Destination'] == row['id']].copy()
            in_df['Distance_Flow_Product'] = (in_df['Travel Distance (km)'] + vkt_offset) * in_df['Flow']

            inflow_sum = in_df['Flow'].sum()
            distance_flow_product_sum_in = in_df['Distance_Flow_Product'].sum()

            n_inflows.append(inflow_sum)
            fkt_inflows.append(distance_flow_product_sum_in)
            if inflow_sum != 0:
                vkt_inflows.append(distance_flow_product_sum_in / inflow_sum)
            else:
                vkt_inflows.append(0)

        # Add a new column with values from the list
        self._aggregated_zone_metrics['n_inflows'] = n_inflows
        self._aggregated_zone_metrics['fkt_outflows'] = fkt_outflows
        self._aggregated_zone_metrics['fkt_inflows'] = fkt_inflows
        self._aggregated_zone_metrics['vkt_outflows'] = vkt_outflows
        self._aggregated_zone_metrics['vkt_inflows'] = vkt_inflows

        # Print
        print(f"\t > Total vehicle-km (by road): {self.aggregated_zone_metrics['fkt_outflows'].sum()} km")
        print(f"\t > Average distance travelled per vehicle (by road): {self.aggregated_zone_metrics['fkt_outflows'].sum() / self.aggregated_zone_metrics['n_outflows'].sum()} km")

    def _get_travel_data(self, df: pd.DataFrame, ors_key: str, road_to_euclidian_ratio: float, batch_size: int) -> pd.DataFrame:
        """
        Get road distances and travel times between traffic zones using ORS or road_to_euclidian_ratio.

        Parameters:
            df (pd.DataFrame): DataFrame containing traffic zones with their geometric centers and IDs.
            ors_key (str): OpenRouteService API key for distance calculations.
            road_to_euclidian_ratio (float): Ratio to convert Euclidean distance to road distance if necessary.
            batch_size (int): Number of origins/destinations per ORS request.

        Returns:
            pd.DataFrame: DataFrame containing the flow data with 'Origin', 'Destination', 'Flow', 'Travel Time (min)', 
                          'Travel Distance (km)', and 'Centroid Distance (km)', with rows where 'Origin' equals 'Destination' removed.
        """       
        # Extract coordinates of the centroid
        coordinates = [[coord[1], coord[0]] for coord in df['geometric_center']]
        num_coordinates = len(coordinates)

        # Initialize the full matrices
        durations = np.zeros((num_coordinates, num_coordinates))
        distances = np.zeros((num_coordinates, num_coordinates))

        # If the user provides an ORS key
        if ors_key is not None:
            print(f"INFO \t Routing using ORS matrix request")

            # Check if the number of coordinates exceeds the batch size
            if num_coordinates > batch_size:
                print(f"ALERT \t {num_coordinates} origins/destinations: this number is greater than the batch size set at {batch_size}. Multiple ORS requests are needed.")

            # Initialize ORS client
            client = openrouteservice.Client(key=ors_key)  # Replace with your ORS API key

            # Split the coordinates into manageable batches
            coordinate_batches = [coordinates[i:i + batch_size] for i in range(0, len(coordinates), batch_size)]

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

                flows.append(0.0)  # Placeholder for the flow for all origin-destination pairs                

                # Calculate euclidean travel distance and append data
                point1 = df.loc[df['id'] == origin_id, 'geometric_center'].iloc[0]
                point2 = df.loc[df['id'] == destination_id, 'geometric_center'].iloc[0]

                euclidian_distance = geodesic(point1, point2).kilometers
                travel_distances_euclidian.append(euclidian_distance)

                # For all zero distances (the case for all combinations if ORS calculation has not been performed)
                if (math.isnan(distances[i][j]) or distances[i][j] == .0) and (origin_id != destination_id):
                    distance = euclidian_distance * road_to_euclidian_ratio
                    travel_distances.append(distance)
                    travel_times.append(distance / 30 * 60)  # Convert to time

                    # If ORS calculation was done, calculate the number of unresolved locations
                    if ors_key is not None:
                        ors_errors += 1
                else:
                    travel_distances.append(distances[i][j] / 1000)  # Convert meters to kilometers
                    travel_times.append(durations[i][j] / 60)  # Convert seconds to minutes

        if ors_errors:
            print(f"ALERT \t ORS was unable to calculate distance or resolve {ors_errors} routes. Road to euclidian distance ratio used instead with a travel speed of 30 km/h. This could affect the model reliability!")

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

        # Remove rows where 'Origin' is equal to 'Destination'
        flows_df = flows_df[flows_df['Origin'] != flows_df['Destination']]
        
        return flows_df

    def _get_attraction_cost_data(self, df:pd.DataFrame, origin_id: str, origin_rows:pd.DataFrame, attraction_feature: str, cost_feature: str) -> tuple:
        """
        Retrieve the attraction and cost data for the given origin.

        Args:
            df (pd.DataFrame): The DataFrame containing traffic zone data.
            origin_id (str): The ID of the origin zone.
            origin_rows (DataFrame): DataFrame rows for the current origin.
            attraction_feature (str): The feature used for attraction (e.g., 'population', 'workplaces').
            cost_feature (str): The feature used for cost (e.g., 'distance_road', 'time_road').

        Returns:
            tuple: Attraction value for origin, list of destination attractions, list of costs.
        """
        if attraction_feature == 'population':
            att_origin = df.loc[df['id'] == origin_id, 'n_people'].values[0]
            dest_att_list = origin_rows['Destination'].apply(lambda x: df.loc[df['id'] == x, 'n_people'].values[0]).tolist()          
        elif attraction_feature == 'workplaces':
            att_origin = df.loc[df['id'] == origin_id, 'n_workplaces'].values[0]
            dest_att_list = origin_rows['Destination'].apply(lambda x: df.loc[df['id'] == x, 'n_workplaces'].values[0]).tolist()
        else:
            raise ValueError(f"ERROR \t Attraction feature is unknown.")

        if cost_feature == 'distance_road':
            cost_list = origin_rows['Travel Distance (km)'].tolist()
        elif cost_feature == 'time_road':
            cost_list = origin_rows['Travel Time (min)'].tolist()
        elif cost_feature == 'distance_centroid':
            cost_list = origin_rows['Centroid Distance (km)'].tolist()
        else:
            raise ValueError(f"ERROR \t Cost feature is unknown.")

        return att_origin, dest_att_list, cost_list

    def _apply_spatial_interaction_model(self, model: str, n_outflows: int, att_origin: float, dest_att_list: list, cost_list: list) -> list:
        """
        Apply the spatial interaction model to calculate flows.

        Args:
            model (str): The spatial interaction model to use.
            n_outflows (int): The number of outgoing trips.
            att_origin (float): The attraction value for the origin.
            dest_att_list (list): The list of attractions for each destination.
            cost_list (list): The list of costs associated with each destination.

        Returns:
            list: Calculated flows based on the model.
        """
        # Use a dictionary on the instance to store persistent flags
        if not hasattr(self, '_apply_spatial_interaction_model_flags'):
            self._apply_spatial_interaction_model_flags = {'beta_info': False, 'unit_surface_alert': False}

        flags = self._apply_spatial_interaction_model_flags  # Shorten access to the flags


        if model == 'gravity_power_1':
            return self.prod_constrained_gravity_power(
                origin_n_trips=n_outflows,
                dest_attractivity_list=dest_att_list,
                cost_list=cost_list,
                gamma=1)
        elif model == 'gravity_exp_1':
            return self.prod_constrained_gravity_exp(
                origin_n_trips=n_outflows,
                dest_attractivity_list=dest_att_list,
                cost_list=cost_list,
                beta=1)
        elif model == 'gravity_exp_01':
            return self.prod_constrained_gravity_exp(
                origin_n_trips=n_outflows,
                dest_attractivity_list=dest_att_list,
                cost_list=cost_list,
                beta=0.1)
        elif model == 'gravity_exp_scaled':
            unit_surface_km2 = self.region.average_zone_area_km2()
            beta = 0.3 * unit_surface_km2**(-0.18) # c.f. Lenormand et al 2016            

            # Display the beta info once
            if not flags['beta_info']:
                print(f"\t > Auto-scaled free parameter based on an average zone area of {unit_surface_km2} km²: Beta = {beta}")
                flags['beta_info'] = True

            # Display the unit surface area alert once
            if unit_surface_km2 < 5 and not flags['unit_surface_alert']:
                print("ALERT \t The average unit surface area is less than 5 km2, which may cause the scaling law of the gravity model to be invalid.")
                flags['unit_surface_alert'] = True

            return self.prod_constrained_gravity_exp(
                origin_n_trips=n_outflows,
                dest_attractivity_list=dest_att_list,
                cost_list=cost_list,
                beta=beta)
        elif model == 'radiation':
            return self.prod_constrained_radiation(
                origin_n_trips=n_outflows,
                origin_attractivity=att_origin,
                dest_attractivity_list=dest_att_list,
                cost_list=cost_list)
        else:
            raise ValueError(f"ERROR \t Spatial interaction model '{model}' is unknown.")

    # Spatial interaction models

    def prod_constrained_gravity_exp(self, origin_n_trips: float, 
                                  dest_attractivity_list: list[float], 
                                  cost_list: list[float], 
                                  beta: float) -> np.ndarray:
        r"""Estimates flows from origin to destinations using a production-constrained gravity model with exponential cost.

        This is a special case of a spatial interaction model, where the cost function$f(d_{ij})$ follows an exponential 
        law.

        .. math::
            f(d_{ij})=e^{-d_{ij} \beta}

        See prod_constrained_gravity_power() for more details on spatial interaction models for trip distribution.

        Args:
            origin_n_trips (float): The total number of trips originating from the origin.
            dest_attractivity_list (list[float]): A list of attractivity values for each destination.
            cost_list (list[float]): A list of cost values corresponding to each destination.
            beta (float, optional): The exponent applied to the cost in the calculation. 

        Returns:
            np.ndarray: An array of estimated flows to each destination.
        """
        flows = np.zeros(len(dest_attractivity_list))
        norm_constant = .0

        # Calculating raw flows and normalization constant
        for j in range(len(flows)):
            if cost_list[j] == 0:
                print(f"ALERT \t Cost function is NULL for some fluxes, setting the corresponding flux to zero", end='\r')
                flows[j] = 0
                attractivity_over_cost = 0
            else:
                attractivity_over_cost = dest_attractivity_list[j] / math.exp(beta * cost_list[j])
                flows[j] = origin_n_trips * attractivity_over_cost

            norm_constant += attractivity_over_cost

        # Normalization
        flows = flows / norm_constant

        return flows

    def prod_constrained_radius(self, origin_n_trips: float, 
                                dest_attractivity_list: list[float], 
                                cost_list: list[float], 
                                radius: float = 10) -> np.ndarray:
        """Estimates flows from origin to destinations using a production-constrained model based on distance radius.

        This is a simplified example of a spatial interaction model. In this case, the product of attractiveness 
        and cost is set to 1 if the destination falls within a specified radius, and 0 otherwise. This function 
        is purely hypothetical and has no practical application justification. It can serve as a test function 
        since it directs all flows to destinations within the given radius.

        See prod_constrained_gravity_power() for more details on spatial interaction models for trip distribution.

        Args:
            origin_n_trips (float): The total number of trips originating from the origin.
            dest_attractivity_list (list[float]): A list of attractivity values for each destination.
            cost_list (list[float]): A list of cost values corresponding to each destination.
            radius (float, optional): The maximum distance for attractivity. 

        Returns:
            np.ndarray: An array of estimated flows to each destination.
        """
        flows = np.zeros(len(dest_attractivity_list))
        norm_constant = .0

        # Calculating raw flows and normalization constant
        for j in range(len(flows)):
            if cost_list[j] == 0:
                print(f"ALERT \t Cost function is NULL for some fluxes, setting the corresponding flux to zero", end='\r')
                flows[j] = 0
                attractivity_over_cost = 0
            else:
                if cost_list[j] <= radius:
                    attractivity_over_cost = 1
                else:
                    attractivity_over_cost = 0
                flows[j] = origin_n_trips * attractivity_over_cost

            norm_constant += attractivity_over_cost

        # Normalization
        flows = flows / norm_constant

        return flows

    def prod_constrained_radiation(self, origin_n_trips: float, 
                                   origin_attractivity: float, 
                                   dest_attractivity_list: list[float], 
                                   cost_list: list[float]) -> np.ndarray:
        r"""Estimates flows from origin to destinations using a production-constrained radiation model.

        This is a special case of an intervening opportunity spatial interaction model. In this case, the algorithm is more
        complex than gravity laws, as the cost function $f(d_{ij})$ includes the intervening opportunities located at a distance 
        smaller than $d_{ij}$ 

        .. math::
            f(d_{ij}) = ((A_i+s_{ij}).(A_i+A_j+s_{ij}))^{-1}
        .. math::
            s_{ij} = \sum_{k \neq i,j} A_k, ~ if ~ d_{ik}<d_{ij} 

        with $s_{ij}$ being the intervening opportunities. Refer to prod_constrained_gravity_power() for other symbols.

        To our knowledge, this model has been initially described in the following publication:
        Simini, F., González, M., Maritan, A. et al. A universal model for mobility and migration patterns. Nature 
        484, 96–100 (2012). https://doi.org/10.1038/nature10856.

        While this model has the advantage of being parameter-free, it seems to not perform well at smaller scales.

        Args:
            origin_n_trips (float): The total number of trips originating from the origin.
            origin_attractivity (float): The attractivity of the origin.
            dest_attractivity_list (list[float]): A list of attractivity values for each destination.
            cost_list (list[float]): A list of cost values corresponding to each destination.

        Returns:
            np.ndarray: An array of estimated flows to each destination.
        """
        # Step 1: Initialize variables
        flows = np.zeros(len(dest_attractivity_list))
        norm_constant = .0
        intervening_opportunity = .0

        # Step 2: Create list of tuples to store initial order of cost_list and order them by distance
        indexed_cost_list = list(enumerate(cost_list))

        # Sort the list of tuples based on the values
        sorted_indexed_cost_list = sorted(indexed_cost_list, key=lambda x: x[1])

        # Prepare a list to store flows with their original indices
        calculated_indexed_flows = []

        # Step 3: Calculate raw flows and normalization constant
        i = 0
        for original_index, _ in sorted_indexed_cost_list:
            j = original_index  # Use the original index from the sorted list

            num = dest_attractivity_list[j]  # Origin attractivity cancels out when normalizing.
            den = (origin_attractivity + intervening_opportunity) * (origin_attractivity + dest_attractivity_list[j] + intervening_opportunity)

            if den == 0:
                print(f"ALERT \t Denominator of radiation model is NULL. Flow is set to zero.", end='\r')
                attractivity_over_cost = 0
                calculated_flow = 0
            else:
                attractivity_over_cost = num / den
                calculated_flow = origin_n_trips * attractivity_over_cost

            flows[j] = calculated_flow

            # Add intervening opportunity only if the next destination is farther away than the current one
            if i >= 1 and (sorted_indexed_cost_list[i - 1] < sorted_indexed_cost_list[i]):
                intervening_opportunity += dest_attractivity_list[j]

            norm_constant += attractivity_over_cost

            # Append the calculated flow with its original index to the list
            calculated_indexed_flows.append((original_index, flows[j]))

            i += 1

        # Step 4: Sort back to the original order using the original indices
        original_order_flows = sorted(calculated_indexed_flows, key=lambda x: x[0])

        # Extract the values from the tuples to get the final list
        final_flows = [value for index, value in original_order_flows]

        # Step 5: Normalize the flows
        final_flows = np.array(final_flows) / norm_constant

        return final_flows

    # Overloading sum 

    def __add__(self, other):
        print("***********************************************")        
        print(f"INFO \t Summing two MobilitySimulator objects")
        print(f"\t Warning: This method implementation is not fully mature and may not handle all edge cases. Note that summing only concerns the results, input properties might make no sense anymore.")        

        if not isinstance(other, MobilitySimulator):
            return NotImplemented
        if self.trip_distribution_params['distance_offset_km'] != .0 or other.trip_distribution_params['distance_offset_km'] != .0:
            print("ERROR \t Cannot handle the case where 'distance_offset_km' is not zero.")
            return NotImplemented
        
        # Sum flows
        if self._flows is not None and other._flows is not None:
            # Concatenate the flows
            combined_flows = pd.concat([self._flows, other._flows], ignore_index=True)
            
            # Group by Origin and Destination and sum the Flow column
            combined_flows = combined_flows.groupby(['Origin', 'Destination'], as_index=False).agg({
                'Flow': 'sum',  # Sum the Flow column
                'Travel Time (min)': 'first',  # Keep first or customize as needed
                'Travel Distance (km)': 'first',  # Keep first or customize as needed
                'Centroid Distance (km)': 'first'  # Keep first or customize as needed
            })
        else:
            ValueError("Run the two simulations before adding two objects.")

        # Create a new MobilitySimulator instance for the result
        new_simulator = MobilitySimulator(
            vehicle_fleet=self.vehicle_fleet,
            region=self.region,
            vehicle_allocation_params=self.vehicle_allocation_params,
            trip_distribution_params=self.trip_distribution_params
        )

        # Assign the combined flows
        new_simulator._flows = combined_flows

        # Sum total vehicles from the vehicle_fleet of both instances
        total_vehicles = self.vehicle_fleet.total_vehicles + other.vehicle_fleet.total_vehicles
        new_simulator.vehicle_fleet.total_vehicles = total_vehicles  # Update the total vehicles in the new instance
        
        # Initialize the aggregated metrics and update
        new_simulator._aggregated_zone_metrics = self._aggregated_zone_metrics
        new_simulator.update_aggregated_metrics()

        # Print
        print(f"\t > Total vehicle-km (by road): {new_simulator.aggregated_zone_metrics['fkt_outflows'].sum()} km")
        print(f"\t > Average distance travelled per vehicle (by road): {new_simulator.aggregated_zone_metrics['fkt_outflows'].sum() / new_simulator.aggregated_zone_metrics['n_outflows'].sum()} km")
        print("***********************************************")  
        return new_simulator

    def update_aggregated_metrics(self):
        """
        Updates the aggregated zone metrics based on flows data.
        """
        if self._flows is None:
            print("Flows or aggregated metrics are not initialized.")
            return
        
        # Loop through each zone ID in the aggregated metrics
        for index, row in self._aggregated_zone_metrics.iterrows():
            zone_id = row['id']

            # Calculate inflows (flows into the zone)
            inflows = self._flows[self._flows['Destination'] == zone_id]['Flow'].sum()

            # Calculate outflows (flows out of the zone)
            outflows = self._flows[self._flows['Origin'] == zone_id]['Flow'].sum()

            # Update inflows and outflows counts
            self._aggregated_zone_metrics.at[index, 'n_inflows'] = inflows
            self._aggregated_zone_metrics.at[index, 'n_outflows'] = round(outflows)

            # Calculate total kilometers (fkt)
            total_km_outflows = (self._flows[self._flows['Origin'] == zone_id]['Travel Distance (km)'] *
                                 self._flows[self._flows['Origin'] == zone_id]['Flow']).sum()
            total_km_inflows = (self._flows[self._flows['Destination'] == zone_id]['Travel Distance (km)'] *
                                self._flows[self._flows['Destination'] == zone_id]['Flow']).sum()

            # Update total kilometers
            self._aggregated_zone_metrics.at[index, 'fkt_outflows'] = total_km_outflows 
            self._aggregated_zone_metrics.at[index, 'fkt_inflows'] = total_km_inflows

            # Calculate average distance (vkt)
            if outflows > 0:
                self._aggregated_zone_metrics.at[index, 'vkt_outflows'] = total_km_outflows / outflows
            else:
                self._aggregated_zone_metrics.at[index, 'vkt_outflows'] = 0

            if inflows > 0:
                self._aggregated_zone_metrics.at[index, 'vkt_inflows'] = total_km_inflows / inflows
            else:
                self._aggregated_zone_metrics.at[index, 'vkt_inflows'] = 0

    # Export and visualization

    def to_csv(self, filepath: str) -> None:
        """
        Saves two CSV files with the main output data for flows and aggregated zone metrics.

        Args:
            filepath (str): Base path for the output files. Automatically appends suffixes
                            "_flows" and "_aggregated_zone_metrics" before ".csv".
        """
        # Remove any existing file extension
        filepath_without_ext, _ = os.path.splitext(filepath)

        # Save each dataframe with the respective suffix
        self._flows.to_csv(f"{filepath_without_ext}_flows.csv")
        self._aggregated_zone_metrics.to_csv(f"{filepath_without_ext}_aggregated_zone_metrics.csv")

    def vehicle_allocation_to_map(self, filepath: str):
        df = self.aggregated_zone_metrics

        # Create the base map with admin boundaries
        m = hlp.create_base_map(self.region)

        # Add TAZ boundaries
        hlp.add_taz_boundaries(m, self.region.traffic_zones)

        # Add number of outflows
        hlp.add_colormapped_feature_group(m, df, self.region, 'n_outflows', 'Number of trips', 'Trips')

        # Add Layer Control and Save
        folium.LayerControl().add_to(m)
        m.save(filepath)

    def trip_distribution_to_map(self, filepath: str, trip_id: int):
        # Create the base map with admin boundaries
        m = hlp.create_base_map(self.region)

        # Add TAZ boundaries
        hlp.add_taz_boundaries(m, self.region.traffic_zones)

        # Add trip distribution flows
        group = self.flows[self.flows['Origin'] == trip_id]
        if not group.empty:
            hlp.add_colormapped_feature_group(m, group, self.region, 'Flow', f'Number of trips from {trip_id}', 'Flows', zone_id_col='Destination')

        folium.LayerControl().add_to(m)
        m.save(filepath)
