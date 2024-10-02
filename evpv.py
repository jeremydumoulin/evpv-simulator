# coding: utf-8

import sys
import os
import importlib.util
import time

from evpv.evcalculator import EVCalculator
from evpv.pvcalculator import PVCalculator
from evpv.evpvsynergies import EVPVSynergies

def main():
    # Welcome message
    print("------------------------------------------------")
    print("         Welcome to the EV-PV Model!")
    print("------------------------------------------------")
    
    print("------------------------------------------------")
    print("Make sure to configure your case study in the config file.")
    print("Let's analyse EV charging and PV production!")
    print("------------------------------------------------")

    print("")

    config_path = input("Enter the path to the python configuration file: ")  # e.g., '/path/to/config.py'
    
    # Dynamically load the config
    config = load_config(config_path)    

    # Run the simulation using the loaded config module
    run_simulation(config)

def run_simulation(config):
    ######################
    ####### Message ######
    ######################

    print("")
    print("------------------------------------------------")
    print(f"Starting the run of the {config.scenario_name} case study")
    print(f"Results are stored in the following folder: {config.output_folder}")
    print("------------------------------------------------")
    print("")

    # Start time
    start_time = time.time()

    ######################################
    ##### STEP 1: EV Charging Demand #####
    ######################################

    ev = EVCalculator(
        mobility_demand = {
            # Required parameters
            'target_area_geojson': config.target_area_geojson, 
            'population_raster': config.population_raster,  
            'destinations_csv': config.destinations_csv,  
            'trips_per_inhabitant': config.trips_per_inhabitant,  
            'zone_width_km': config.zone_width_km,  
            'ORS_key': config.ORS_key,

            # Optional parameters            
            'road_to_euclidian_ratio': config.road_to_euclidian_ratio, 
            'target_area_extension_km': config.target_area_extension_km, 
            'population_to_ignore_share': config.population_to_ignore_share, 
            'spatial_interaction_model': config.spatial_interaction_model, 
            'attraction_feature': config.attraction_feature, 
            'cost_feature': config.cost_feature, 
            'km_per_capita_offset': config.km_per_capita_offset       
        },
        ev_fleet = config.ev_fleet(), # Warning, this is a function !
        charging_efficiency = config.charging_efficiency,
        charging_scenario = {
            # Required
            'Home': config.charging_scenario['Home'],
            'Destination': config.charging_scenario['Destination'],

            # Optional
            'travel_time_origin_destination_h': config.travel_time_origin_destination_h, 
            'time_step_h': config.time_step_h, 
            }  
        )

    # Run the EV demand simulation based on the provided parameters
    ev.compute_ev_demand()

    # Save the EV charging demand results to the output folder
    ev.save_results(output_folder = config.output_folder, prefix = config.scenario_name)

    ######################################
    ######## STEP 2: PV Production #######
    ######################################

    pv = PVCalculator(
        environment = {
            'latitude': config.latitude, 
            'longitude': config.longitude,  
            'year': config.year
            }, 
        pv_module = {
            'efficiency': config.efficiency,

            # Optional
            'temperature_coefficient': config.temperature_coefficient 
            }, 
        installation = {
            'type': config.installation,
            # Optional
            'system_losses': config.system_losses
        })

    # Run the PV simulation for the specified parameters
    pv.compute_pv_production()

    # Save the PV production results to a CSV file in the output folder
    pv.results.to_csv(f"{config.output_folder}/{config.scenario_name}_PVproduction.csv")

    ######################################
    ####### STEP 3: EV-PV Synergies ######
    ######################################

    evpv = EVPVSynergies(pv_calculator = pv, ev_calculator = ev, pv_capacity_MW = config.pv_capacity_MW)

    # Compute all synergy metrics over a given time period (January 1st to January 30th).
    # This will include metrics such as energy coverage ratio, self-consumption ratio, and others.
    daily_kpis = evpv.daily_metrics(start_date = config.start_date, end_date = config.end_date)

    # Save the calculated daily synergy metrics (KPI values) to a CSV file in the output folder
    daily_kpis.to_csv(f"{config.output_folder}/{config.scenario_name}_EVPV_KPIs.csv")

    ######################
    ####### Message ######
    ######################

    # End time
    end_time = time.time()

    # Calculate the duration
    duration = end_time - start_time
    minutes = int(duration // 60)  # Get the whole minutes
    seconds = duration % 60         # Get the remaining seconds

    print("")
    print("")
    print("------------------------------------------------")
    print(f"Simulation completed")
    print(f"Elapsed time: : {minutes} minutes and {seconds:.2f} seconds")
    print("------------------------------------------------")

def load_config(config_path):
    # Ensure the provided config file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Get the module name (e.g., 'config') from the file name (e.g., 'config.py')
    config_name = os.path.splitext(os.path.basename(config_path))[0]

    # Dynamically load the config module
    spec = importlib.util.spec_from_file_location(config_name, config_path)
    config_module = importlib.util.module_from_spec(spec)
    sys.modules[config_name] = config_module
    spec.loader.exec_module(config_module)

    return config_module

if __name__ == "__main__":
    main()
