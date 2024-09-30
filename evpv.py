# coding: utf-8

import sys
import os
import argparse
import ast  # Import the ast module for safer evaluation

from evpv.evcalculator import EVCalculator
from evpv.pvcalculator import PVCalculator
from evpv.evpvsynergies import EVPVSynergies

def run_simulation(config):
    ######################################
    ##### STEP 0: Allocate ev fleet ######
    ######################################
    for ev in config['ev_fleet']:
        print("hello")

    ######################################
    ##### STEP 1: EV Charging Demand #####
    ######################################

    ev = EVCalculator(
        mobility_demand = {
            # Required parameters
            'target_area_geojson': config['target_area_geojson'],  # Geographic area of interest (Addis Ababa)
            'population_raster': config['population_raster'],  # Population data for the area
            'destinations_csv': config['destinations_csv'],  # Key destinations for mobility patterns (e.g., workplaces)
            'trips_per_inhabitant': config['trips_per_inhabitant'],  # Number of trips per person per day
            'zone_width_km': config['zone_width_km'],  # Resolution of geographic zones for mobility demand calculations
            'ORS_key': config['ORS_key'],  # OpenRouteService API key (optional for more accurate routing)
        },
        ev_fleet = [
            [EVCalculator.preset['car'], 1.0],  # 100% cars in the EV fleet 
            [EVCalculator.preset['motorbike'], 0.0]  # 0% motorbikes in the fleet
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

def load_config_from_file(config_path):
    config = {}

    # Reading the config file line by line
    with open(config_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue  # Skip empty lines and comments
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()
            try:
                # Use ast.literal_eval for safe evaluation of Python literals
                config[key] = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                # If there's an error in evaluating, just keep the raw string value
                config[key] = value

    return config

def main():
    print("Welcome to the EV-PV Configuration Loader!")

    # Command-line argument parser
    parser = argparse.ArgumentParser(description="EV-PV Simulation Tool")
    parser.add_argument(
        "-c", "--config", 
        type=str, 
        help="Path to the configuration file"
    )

    # Parse command-line arguments
    args = parser.parse_args()

    # If config file is not provided, ask the user for input
    if not args.config:
        config_path = input("Please provide the path to the configuration file (e.g., config.txt): ")
    else:
        config_path = args.config

    # Validate the provided config path
    if not os.path.isfile(config_path):
        print(f"Error: The file '{config_path}' does not exist.")
        sys.exit(1)

    # Load the configuration file
    config = load_config_from_file(config_path)

    # Run the simulation using the loaded config
    run_simulation(config)

if __name__ == "__main__":
    main()
