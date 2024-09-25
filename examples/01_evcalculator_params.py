# coding: utf-8

""" 
A python script illustrating the usage of the EVCalculator class, notably showing all required/optional parameters 
All the optional parameters are populated with the default inputs
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Append parent directory to include evpv

from evpv.evcalculator import EVCalculator

######################################
###### Scenario Name (Optional) ######
######################################

sc = '01'

######################################
######## EV Charging Demand ##########
######################################

ev = EVCalculator(
    # Attributes for mobility demand simulation
    mobility_demand = {
        # REQUIRED
        'target_area_geojson': 'input/gadm41_ETH_1_AddisAbeba.json', # Path to the geojson file containing the shape of the target area (or region of interest)
        'population_raster': 'input/GHS_POP_merged_4326_3ss_V1_0_R8andR9_C22_cropped.tif', # Path to raster file (.tif) with the population density
        'destinations_csv': 'input/workplaces.csv', # Path to the csv file with the list of potential destinations
        'trips_per_inhabitant': 0.1, # Average number of trips per inhabitant (from origin to destination, e.g., home to Destination)
        'zone_width_km': 5, # Target width (in km) of the zones (i.e., spatial resolution) - will be slighlty adapted by the algorithm

        # OPTIONAL
        'ORS_key': None, #Open Route Service (ORS) API key. If no key is provided, the distance by road is estimated using an empirical ratio set by the user
        'road_to_euclidian_ratio': 1.63, # Empirical ratio between the distance by road and the euclidian distance (distance as the crow flies) 
        'target_area_extension_km': 0.0, # Extension of the the target area to include also in- and out- flows from outside
        'population_to_ignore_share': 0.0, # Share of the population to ignore (will speed up calculation by ignoring sparsely populated zones)
        'spatial_interaction_model': 'gravity_exp_scaled', # Type of spatial interaction model to use ('gravity_exp_scaled' = autocalibrated gravity model)
        'attraction_feature': 'destinations', # Attraction feature used in the spatial interaction model ('destinations', 'population')
        'cost_feature': 'distance_road', # Cost feature used in the spatial interaction model ('distance_road', 'distance_centroid', 'time_road')
        'km_per_capita_offset': 0.0, # Additionnal daily distance travelled (in km) from the origin to destination (one way)  
    },

    # EV Fleet in the form [[vehicle1, share1], [vehicle2, share2], ...]
    ev_fleet = [
        [EVCalculator.preset['car'], 1.0], 
        [EVCalculator.preset['motorbike'], 0.0]
    ],

    # Charging efficiency between 0 and 1
    charging_efficiency = 0.9,

    # Charging scenario 
    charging_scenario = {
        # REQUIRED
        "Home": {
            "Share": 0.0, # Share of vehicles charging at home
            "Arrival time": [18, 2], # Arrival time at home in hours [average, standard_deviation]
            "Smart charging": 0.0 # Share of vehicles with smart charging at home
        },
        "Destination": {
            "Share": 1.0, # Share of vehicles charging at destination
            "Arrival time": [9, 2], # Arrival time at destination in hours [average, standard_deviation]
            "Smart charging": 0.0  # Share of vehicles with smart charging at destination
        },

        # OPTIONAL
        'travel_time_origin_destination_h':  0.5, # Average travel time (in hours) form origin to/from destination (used for smart charging)
        'time_step_h': 0.1, # Time step (in hours) for the charging curve
        }
    )

# Run the EV demand simulation
ev.compute_ev_demand()

# Save the results
ev.save_results(output_folder = "output", prefix = sc)