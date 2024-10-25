# coding: utf-8

""" 
A python script illustrating how to specify your own mix of electric vehicles and charger powers
The script first defines a given mix, populates a EVCalculator object with it, and then calculates
the EV charging demand
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Append parent directory to include evpv modules

from evpv.evcalculator import EVCalculator
from evpv.pvcalculator import PVCalculator
from evpv.evpvsynergies import EVPVSynergies

######################################
## Prefix for results (for clarity) ##
######################################

# A prefix used to distinguish outputs in the output folder (e.g., "00")
pr = '03'

######################################
### EVs and charger powers to use ####
######################################

# Nissan Leaf 
nissan_leaf = {
    'ev_consumption': 0.166,  # Electric vehicle consumption (kWh/km)
    'battery_capacity': 40, # Battery capacity (kWh) - Importan - Note that the capacity will be reduced in the calculation (useful capacity, typically 80% of the nominal capacity, see Pareschi et al., Applied Energy, 2020))
    'charger_power': {
        'Origin': [[3.6, 0.4], [7, 0.4], [11, 0.2]],  # Mix of charger power at origin (home)
        'Destination': [[3.6, 0.4], [7, 0.4], [11, 0.2]],  # Mix of charger power at destination (workplaces)
        'Intermediate': [[3.6, 0.4], [7, 0.4], [11, 0.2]]  # Mix of charger power at intermediate stops (POIs)        
    }
}

# Tesla Model 3 
tesla_model_3 = {
    'ev_consumption': 0.144,  # Electric vehicle consumption (kWh/km)
    'battery_capacity': 75, # Battery capacity (kWh)
    'charger_power': {
        'Origin': [[7, 0.6], [11, 0.3], [22, 0.1]],  # Mix of charger power at origin (home)
        'Destination': [[7, 0.6], [11, 0.3], [22, 0.1]],  # Mix of charger power at destination (workplaces)
        'Intermediate': [[7, 0.6], [11, 0.3], [22, 0.1]]  # Mix of charger power at intermediate stops (POIs)           
    }
}

# Renault Zoe 
renault_zoe = {
    'ev_consumption': 0.132,  # Electric vehicle consumption (kWh/km)
    'battery_capacity': 52, # Battery capacity (kWh)
    'charger_power': {
        'Origin': [[3.6, 0.5], [7, 0.3], [22, 0.2]],  # Mix of charger power at origin (home)
        'Destination': [[3.6, 0.5], [7, 0.3], [22, 0.2]],  # Mix of charger power at destination (workplaces)
        'Intermediate': [[3.6, 0.5], [7, 0.3], [22, 0.2]]  # Mix of charger power at intermediate stops (POIs)
    }
}

######################################
######### EV Charging Demand #########
######################################

# This step calculates the EV charging demand based on mobility patterns for a specific geographic area,
# population, and a predefined electric vehicle fleet.

# Create the EVCalculator object, specifying mobility demand parameters such as the target area, population,
# and number of trips per person. The 'ev_fleet' parameter describes the fleet of vehicles being considered.
ev = EVCalculator(
    mobility_demand = {
        # Required parameters
        'target_area_geojson': 'input/gadm41_ETH_1_AddisAbeba.json',  # Geographic area of interest (Addis Ababa)
        'population_raster': 'input/GHS_POP_merged_4326_3ss_V1_0_R8andR9_C22_cropped.tif',  # Population data for the area
        'destinations_csv': 'input/workplaces.csv',  # Key destinations for mobility patterns (e.g., workplaces)
        'intermediate_stops_csv': 'input/intermediate_stops.csv',  # Key intermediate stops for mobility patterns (e.g., POIs)
        'trips_per_inhabitant': 0.1,  # Number of trips per person per day
        'zone_width_km': 5,  # Resolution of geographic zones for mobility demand calculations

        # Optional parameters
        'ORS_key': None,  # OpenRouteService API key (optional for more accurate routing)
    },
    ev_fleet = [
        [nissan_leaf, 0.5],  
        [tesla_model_3, 0.1],
        [renault_zoe , 0.4]  
    ],
    charging_efficiency = 0.9,  # Efficiency of EV charging
    charging_scenario = {
        "Home": {
            "Share": 0.0,  # No charging at home
            "Arrival time": [18, 2],  # Arrival times at home (mean time and std dev)
            "Smart charging": 0.0  # No smart charging at home
        },
        "Destination": {
            "Share": 1.0,  # 100% charging at destinations (e.g., workplaces)
            "Arrival time": [9, 2],  # Arrival times at destinations (mean time and std dev)
            "Smart charging": 0.0  # No smart charging at destination
        },
        "Intermediate": {
            "Share": 0.0, 
            "Smart charging": 0.0 
        }}
    )

# Run the EV demand simulation based on the provided parameters
ev.compute_ev_demand()

# Save the EV charging demand results to the output folder
ev.save_results(output_folder = "output", prefix = pr)