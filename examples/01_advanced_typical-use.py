# coding: utf-8

""" 
A python script illustrating the basic usage of the EV-PV model.
The script calculates electric vehicle (EV) charging demand, photovoltaic (PV) power production,
and evaluates synergies between EV charging and PV generation.

It consists of three main steps:
1. Compute EV charging demand based on mobility demand simulation inputs, electric vehicle fleet information, and charging scenario.
2. Simulate PV power production based on geographic location and PV system parameters.
3. Calculate the synergy metrics (e.g., self-sufficiency, energy coverage) between the EV charging demand and PV generation.
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
pr = '00'

######################################
##### STEP 1: EV Charging Demand #####
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
        'n_vehicles': 1000,  # Number of EVs
        'zone_width_km': 3,  # Resolution of geographic zones for mobility demand calculations
        'ORS_key': None,  # OpenRouteService API key (optional for more accurate routing)
    },
    ev_fleet = [
        [EVCalculator.preset['car'], 1.0],  # Share of default cars in the EV fleet 
        [EVCalculator.preset['motorbike'], 0.0]  # Share of defauly motorbikes in the fleet
    ],
    charging_efficiency = 0.9,  # Efficiency of EV charging
    charging_scenario = {
        "Home": {
            "Share": 0.5,  # No charging at home
            "Arrival time": [18, 3],  # Arrival times at home (mean time and std dev)
            "Smart charging": 0.0  # No smart charging at home
        },
        "Destination": {
            "Share": 0.21,  # 100% charging at destinations (e.g., workplaces)
            "Arrival time": [9, 2],  # Arrival times at destinations (mean time and std dev)
            "Smart charging": 0.0  # No smart charging at destination
        },
        "Intermediate": {
            "Share": 0.29,  
            "Smart charging": 0.0  
        }}
    )

# Run the EV demand simulation based on the provided parameters
ev.compute_ev_demand()

# Save the EV charging demand results to the output folder
ev.save_results(output_folder = "output", prefix = pr)

######################################
######## STEP 2: PV Production #######
######################################

# This step simulates PV power generation based on geographic location, system specifications, and module efficiency.

# Create the PVCalculator object to simulate PV production.
# The 'environment' dictionary specifies the location and year of interest (Addis Ababa in 2020),
# while the 'pv_module' dictionary defines the efficiency of the solar panels, and 'installation' specifies the type of setup.
pv = PVCalculator(
    environment = {
        'latitude': 9.005401,  # Latitude of the location (Addis Ababa)
        'longitude': 38.763611,  # Longitude of the location
        'year': '2020'  # Year of PV production simulation
        }, 
    pv_module = {
        'efficiency': 0.22  # PV panel efficiency
        }, 
    installation = {
        'type': 'groundmounted_fixed'  # Type of PV installation (fixed, ground-mounted)
    })

# Run the PV simulation for the specified parameters
pv.compute_pv_production()

# Save the PV production results to a CSV file in the output folder
pv.results.to_csv(f"output/{pr}_PVproduction.csv")

######################################
####### STEP 3: EV-PV Synergies ######
######################################

# This step evaluates the synergy between the previously computed EV charging demand and PV power generation,
# calculating various metrics such as self-sufficiency ratio, energy coverage, and excess PV generation.

# Create the EVPVSynergies object to compute synergy metrics between EV and PV systems.
# It takes the PV and EV calculators as input along with the installed PV capacity (in MW).
evpv = EVPVSynergies(pv_calculator = pv, ev_calculator = ev, pv_capacity_MW = 280)

# Compute all synergy metrics over a given time period (January 1st to January 30th).
# This will include metrics such as energy coverage ratio, self-consumption ratio, and others.
daily_kpis = evpv.daily_metrics(start_date = '01-01', end_date = '01-30')

# Save the calculated daily synergy metrics (KPI values) to a CSV file in the output folder
daily_kpis.to_csv(f"output/{pr}_EVPV_KPIs.csv")
