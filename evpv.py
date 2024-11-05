# coding: utf-8

import sys
import os
import importlib.util
import time

from evpv.vehicle import Vehicle
from evpv.vehiclefleet import VehicleFleet
from evpv.region import Region
from evpv.mobilitysimulator import MobilitySimulator
from evpv.pvsimulator import PVSimulator
from evpv.chargingsimulator import ChargingSimulator
from evpv.evpvsynergies import EVPVSynergies

# Main

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

# Run the simulation

def run_simulation(config):

    # Welcome message

    print("")
    print("------------------------------------------------")
    print(f"Starting the run of the {config.scenario_name} case study")
    print(f"Results are stored in the following folder: {config.output_folder}")
    print("------------------------------------------------")
    print("")

    # Start time

    start_time = time.time()

    # STEP 1: Define the electric vehicle fleet
    # Create instances of Vehicle representing different types with specific attributes (e.g., battery capacity, consumption rate)

    fleet = create_fleet_from_config(config)

    # STEP 2: Define the region of interest and its traffic zone properties
    # Create a region using geospatial data (region boundaries, population raster, workplaces, points of interest)

    region = Region(
        region_geojson=config.region_geojson,    
        population_raster=config.population_raster,
        workplaces_csv=config.workplaces_csv,
        pois_csv=config.pois_csv,
        traffic_zone_properties={
            "target_size_km": config.target_size_km,
            "shape": config.zone_shape,
            "crop_to_region": config.crop_to_region
        }
    )

    region.to_map(f"{config.output_folder}/{config.scenario_name}_region.html") # Save region map to file

    # STEP 3: Perform mobility simulation using a gravity model for commuting
    # Initialize the MobilitySimulator to simulate commuting patterns based on region and vehicle fleet data

    mobility_sim = MobilitySimulator(
        vehicle_fleet=fleet,
        region=region,
        vehicle_allocation_params={
            "method": config.allocation_method,                
            "randomness": config.randomness
        }, 
        trip_distribution_params={
            "model_type": config.model_type,                
            "attraction_feature": config.attraction_feature,     
            "cost_feature": config.cost_feature,             
            "road_to_euclidian_ratio": config.road_to_euclidian_ratio,         
            "ors_key": config.ors_key,
            "distance_offset_km": config.distance_offset_km                        
        }
    )

    # Allocate vehicles to zones and simulate trips
    mobility_sim.vehicle_allocation()
    mobility_sim.trip_distribution()

    # Save mobility simulation results and generate visualizations
    mobility_sim.to_csv(f"{config.output_folder}/{config.scenario_name}_results.csv")
    mobility_sim.vehicle_allocation_to_map(f"{config.output_folder}/{config.scenario_name}_MobilitySimulation_allocation.html")

    # STEP 4: Charging demand simulation based on EV travel patterns and charging scenarios
    # Define charging options (home, work, points of interest) with corresponding power and arrival time distributions

    charging_sim = ChargingSimulator(
        vehicle_fleet=fleet,
        region=region,
        mobility_demand=mobility_sim,
        charging_efficiency=config.charging_efficiency,
        scenario=config.scenario
    )

    # Compute spatial and temporal charging demand
    charging_sim.compute_spatial_demand()
    charging_sim.compute_temporal_demand(time_step=config.time_step) # Time step in hours

    # Optional: Apply smart charging to reduce peak demand
    # charging_sim.apply_smart_charging(location=["home"], charging_strategy="peak_shaving", share=0.5)

    # Save charging demand data and visualizations
    charging_sim.to_csv(f"{config.output_folder}/{config.scenario_name}_ChargingDemand.csv")
    charging_sim.chargingdemand_total_to_map(f"{config.output_folder}/{config.scenario_name}_ChargingDemand_total.html")
    charging_sim.chargingdemand_pervehicle_to_map(f"{config.output_folder}/{config.scenario_name}_ChargingDemand_pervehicle.html")
    charging_sim.chargingdemand_nvehicles_to_map(f"{config.output_folder}/{config.scenario_name}_ChargingDemand_n_vehicles.html")

    # STEP 5: PV Simulation for calculating photovoltaic power production
    # Initialize PVSimulator using location coordinates, module characteristics, and installation type

    pv = PVSimulator(
        environment={
            'latitude': region.centroid_coords()[0],  
            'longitude': region.centroid_coords()[1],  
            'year': config.year  
        }, 
        pv_module={
            'efficiency': config.efficiency,
            'temperature_coefficient': config.temperature_coefficient 
        }, 
        installation={
            'type': config.installation_type,  
            'system_losses': config.system_losses
        }
    )

    pv.compute_pv_production()  # Calculate PV production based on the defined parameters
    pv.results.to_csv(f"{config.output_folder}/{config.scenario_name}_PVProduction.csv")  # Save PV production data

    # STEP 6: EV-PV Synergy Analysis
    # Calculate synergies between PV generation and EV charging demand over a defined time period

    evpv = EVPVSynergies(pv=pv, ev=charging_sim, pv_capacity_MW=config.pv_capacity_MW)

    # Calculate daily synergy metrics for the first week of January, adjusting recompute_probability as needed
    synergy_metrics = evpv.daily_metrics(config.start_date, config.end_date, recompute_probability=config.recompute_probability)
    synergy_metrics.to_csv(f"{config.output_folder}/{config.scenario_name}_EVPVSynergies.csv") # Save synergy metrics data

    # End time and message 

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

# Helper functions

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

def create_fleet_from_config(config):
    vehicles = []
    for vehicle_data in config.fleet_config["vehicle_types"]:
        vehicle = Vehicle(
            name=vehicle_data["name"],
            battery_capacity_kWh=vehicle_data["battery_capacity_kWh"],
            consumption_kWh_per_km=vehicle_data["consumption_kWh_per_km"],
            max_charging_power_kW=vehicle_data["max_charging_power_kW"]
        )
        vehicles.append((vehicle, vehicle_data["share"]))

    # Instantiate the fleet with the total vehicle count and vehicles with proportions
    fleet = VehicleFleet(
        total_vehicles=config.fleet_config["total_vehicles"],
        vehicle_types=vehicles
    )
    return fleet

if __name__ == "__main__":
    main()
