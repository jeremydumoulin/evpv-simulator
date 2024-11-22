# coding: utf-8

""" 
A python script for a parametric analysis of charging power influence on maximum vehicles charging and peak power.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Append parent directory to include evpv modules

from evpv.vehicle import Vehicle
from evpv.vehiclefleet import VehicleFleet
from evpv.region import Region
from evpv.mobilitysimulator import MobilitySimulator
from evpv.chargingsimulator import ChargingSimulator

# Dummy single-vehicle setup
bev = Vehicle(name="BEV", battery_capacity_kWh=51, consumption_kWh_per_km=0.183)
fleet = VehicleFleet(total_vehicles=100000, vehicle_types=[[bev, 1.0]])

# Create a simple region
region = Region(
    region_geojson="input/gadm41_ETH_1_AddisAbeba.json",
    population_raster="input/GHS_POP_merged_4326_3ss_V1_0_R8andR9_C22_cropped.tif",
    workplaces_csv="input/workplaces.csv",
    pois_csv="input/pois.csv",
    traffic_zone_properties={"target_size_km": 2, "shape": "rectangle", "crop_to_region": True}
)

mobility_sim = MobilitySimulator(
    vehicle_fleet=fleet,
    region=region,
    vehicle_allocation_params={
        "method": "population",                
        "randomness": 0.0
    }, 
    trip_distribution_params={
        "model_type": "gravity_exp_scaled",                
        "attraction_feature": "workplaces",     
        "cost_feature": "distance_centroid",             
        "road_to_euclidian_ratio": 1.48,         
        "ors_key": "5b3ce3597851110001cf6248879c0a16f2754562898e0826e061a1a3",
        "distance_offset_km": 0.0                         
    }
)

# Allocate vehicles to zones and simulate trips
mobility_sim.vehicle_allocation()
mobility_sim.trip_distribution()

# Define charging parameters for the home-only scenario
charging_power_options_kW = [1, 2, 4, 6, 9, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]
results = []

# Run simulations for each charging power
for power in charging_power_options_kW:
    print(f"Running simulation for charging power: {power} kW")
    max_vehicles_list = []
    max_power_list = []

    # Repeat simulation 5 times
    for i in range(5):
        charging_sim = ChargingSimulator(
            vehicle_fleet=fleet,
            region=region,
            mobility_demand=mobility_sim,
            charging_efficiency=0.9,
            scenario={
                'home': {
                    'share': 1.0,  # 100% home charging
                    'power_options_kW': [[power, 1.0]],
                    'arrival_time_h': [9, 1.8]
                },
                'work': {
                    'share': 0.0,
                    'power_options_kW': [[power, 1.0]],
                    'arrival_time_h': [18, 1.0]
                },
                'poi': {
                    'share': 0.0,
                    'power_options_kW': [[power, 1.0]],
                    'arrival_time_h': [18, 1.0]
                }
            }
        )

        charging_sim.compute_spatial_demand()
        charging_sim.compute_temporal_demand(time_step=1 / 20)

        # Collect metrics
        max_charging_vehicles = charging_sim.temporal_demand_profile_aggregated["total_vehicle_charging"].max()
        max_charging_power = charging_sim.temporal_demand_profile_aggregated["total"].max()
        max_vehicles_list.append(max_charging_vehicles)
        max_power_list.append(max_charging_power)

    # Compute mean and standard deviation
    results.append({
        'charging_power_kW': power,
        'max_charging_vehicles_mean': np.mean(max_vehicles_list),
        'max_charging_vehicles_std': np.std(max_vehicles_list),
        'max_charging_power_mean': np.mean(max_power_list),
        'max_charging_power_std': np.std(max_power_list)
    })

# Convert results to DataFrame
results_df = pd.DataFrame(results)
results_df.to_csv("output/parametric_analysis_results.csv", index=False)

# Display results
print(results_df)