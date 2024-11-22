# coding: utf-8

""" 
A python script illustrating the basic usage of the EV-PV model.
The script calculates electric vehicle (EV) charging demand, photovoltaic (PV) power production,
and evaluates synergies between EV charging and PV generation.
"""

import sys
import os
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Append parent directory to include evpv modules

from evpv.vehicle import Vehicle
from evpv.vehiclefleet import VehicleFleet
from evpv.region import Region
from evpv.mobilitysimulator import MobilitySimulator
from evpv.pvsimulator import PVSimulator
from evpv.chargingsimulator import ChargingSimulator
from evpv.evpvsynergies import EVPVSynergies

# STEP 1: Define the region

region = Region(
    region_geojson="input/gadm41_ETH_1_AddisAbeba.json",    
    population_raster="input/GHS_POP_merged_4326_3ss_V1_0_R8andR9_C22_cropped.tif",
    workplaces_csv="input/workplaces.csv",
    pois_csv="input/pois.csv",
    traffic_zone_properties={
        "target_size_km": 2,
        "shape": "rectangle",
        "crop_to_region": True
    }
)

region.to_map("output/region.html") # Save region map to file
region.to_csv("output/region.csv") # Save region map to file


# STEP 2: Define the vehicle fleet

bev = Vehicle(name="BEV", battery_capacity_kWh=60, consumption_kWh_per_km=0.183)
phev = Vehicle(name="PHEV", battery_capacity_kWh=15, consumption_kWh_per_km=0.183, max_charging_power_kW=11) # Define max charging power for motorcycles

# Create a fleet of vehicles, specifying proportions for each vehicle type (e.g., 90% cars, 10% motorcycles)
fleet = VehicleFleet(total_vehicles=100000, vehicle_types=[[bev, 0.8], [phev, 0.2]])

# STEP 3: Perform mobility simulation using a gravity model for commuting
# Initialize the MobilitySimulator to simulate commuting patterns based on region and vehicle fleet data

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
        "ors_key": "5b3ce3597851110001cf6248879c0a16f2754562898e0826e061a1a3", # Optional OpenRouteService API key "5b3ce3597851110001cf6248879c0a16f2754562898e0826e061a1a3"
        "distance_offset_km": 0.0                         
    }
)

# Allocate vehicles to zones and simulate trips
mobility_sim.vehicle_allocation()
mobility_sim.trip_distribution()

# Save mobility simulation results and generate visualizations
mobility_sim.to_csv("output/MobilitySimulation_results.csv")
mobility_sim.vehicle_allocation_to_map("output/MobilitySimulation_allocation.html")
mobility_sim.trip_distribution_to_map("output/MobilitySimulation_flows_example.html", "10_4")

# STEP 4: Charging demand simulation based on EV travel patterns and charging scenarios
# Define charging options (home, work, points of interest) with corresponding power and arrival time distributions

# Liste des scénarios de recharge
scenarios = [
    {'name': 'Home', 'home': 1.0, 'work': 0.0, 'poi': 0.0},
    {'name': 'Work', 'home': 0.0, 'work': 1.0, 'poi': 0.0},
    {'name': 'Mixed', 'home': 0.2, 'work': 0.5, 'poi': 0.3}
]

# Nombre de répétitions pour chaque scénario
repeats = 5

# Dictionnaire pour stocker les résultats
results = {}

for scenario in scenarios:
    scenario_name = scenario['name']
    max_cars_charging = []  # Stocke les valeurs maximales pour chaque répétition

    for _ in range(repeats):
        # Initialisation de la simulation de recharge
        charging_sim = ChargingSimulator(
            vehicle_fleet=fleet,
            region=region,
            mobility_demand=mobility_sim,
            charging_efficiency=0.9,
            scenario={
                'home': {
                    'share': scenario['home'],  
                    'power_options_kW': [[3.2, 0.45], [7.4, 0.4], [11, 0.15]],    
                    'arrival_time_h': [18, 2.7]  # Arrival time with mean and std deviation
                },
                'work': {
                    'share': scenario['work'],  
                    'power_options_kW': [[7.4, 0.25], [11, 0.5], [22, 0.25]],    
                    'arrival_time_h': [9, 1.8]
                },
                'poi': {
                    'share': scenario['poi'],  
                    'power_options_kW': [[7.4, 0.15], [11, 0.15], [22, 0.55], [50, 0.15]]    
                }
            }
        )

        # Calcul de la demande temporelle
        charging_sim.compute_spatial_demand()
        charging_sim.compute_temporal_demand(time_step=1/30)  # Pas de temps en heures

        max_charging = charging_sim.temporal_demand_profile_aggregated["total_vehicle_charging"].max()   # Somme sur les profils pour chaque pas de temps
        max_cars_charging.append(max_charging)

    # Calcul de la moyenne et de l'écart type pour ce scénario
    results[scenario_name] = {
        'mean_max_cars_charging': np.mean(max_cars_charging),
        'std_max_cars_charging': np.std(max_cars_charging)
    }

# Affichage des résultats
for scenario, stats in results.items():
    print(f"Scenario: {scenario}")
    print(f"  Mean max cars charging: {stats['mean_max_cars_charging']:.2f}")
    print(f"  Std dev: {stats['std_max_cars_charging']:.2f}")


# Export des résultats en CSV
results_df = pd.DataFrame(results)
results_df.to_csv("output/charging_scenarios_results.csv", index=False)
