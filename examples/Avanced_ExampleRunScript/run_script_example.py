# coding: utf-8

""" 
A python script illustrating the basic usage of the EV-PV model.
The script calculates electric vehicle (EV) charging demand, photovoltaic (PV) power production,
and evaluates synergies between EV charging and PV generation.

It consists of the following steps:
1. Define the electric vehicle fleet.
2. Define the region of interest and its traffic zone properties.
3. Perform a mobility simulation for commuting between home and work using a gravity model.
4. Simulate EV charging demand at various locations (home, work, points of interest).
5. Simulate PV power production for the region.
6. Calculate synergies between EV charging demand and PV power production.
"""

from evpv.vehicle import Vehicle
from evpv.vehiclefleet import VehicleFleet
from evpv.region import Region
from evpv.mobilitysimulator import MobilitySimulator
from evpv.pvsimulator import PVSimulator
from evpv.chargingsimulator import ChargingSimulator
from evpv.evpvsynergies import EVPVSynergies

# STEP 1: Define the electric vehicle fleet
# Create instances of Vehicle representing different types with specific attributes (e.g., battery capacity, consumption rate)

car = Vehicle(name="car", battery_capacity_kWh=50, consumption_kWh_per_km=0.18)
motorcycle = Vehicle(name="motorcycle", battery_capacity_kWh=10, consumption_kWh_per_km=0.05, max_charging_power_kW=10) # Define max charging power for motorcycles

# Create a fleet of vehicles, specifying proportions for each vehicle type (e.g., 90% cars, 10% motorcycles)
fleet = VehicleFleet(total_vehicles=1000, vehicle_types=[[car, 0.9], [motorcycle, 0.1]])

# STEP 2: Define the region of interest and its traffic zone properties
# Create a region using geospatial data (region boundaries, population raster, workplaces, points of interest)

region = Region(
    region_geojson="input/gadm41_ETH_1_AddisAbeba.json",    
    population_raster="input/GHS_POP_merged_4326_3ss_V1_0_R8andR9_C22_cropped.tif",
    workplaces_csv="input/workplaces.csv",
    pois_csv="input/pois.csv",
    traffic_zone_properties={
        "target_size_km": 5,
        "shape": "rectangle",
        "crop_to_region": True
    }
)

region.to_map("output/00_region.html") # Save region map to file

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
        "cost_feature": "distance_road",             
        "road_to_euclidian_ratio": 1.63,         
        "ors_key": None, # Optional OpenRouteService API key
        "distance_offset_km": 0.0                         
    }
)

# Allocate vehicles to zones and simulate trips
mobility_sim.vehicle_allocation()
mobility_sim.trip_distribution()

# Save mobility simulation results and generate visualizations
mobility_sim.to_csv("output/00_MobilitySimulation_results.csv")
mobility_sim.vehicle_allocation_to_map("output/00_MobilitySimulation_allocation.html")
mobility_sim.trip_distribution_to_map("output/00_MobilitySimulation_flows_example.html", "3_3")

# STEP 4: Charging demand simulation based on EV travel patterns and charging scenarios
# Define charging options (home, work, points of interest) with corresponding power and arrival time distributions

charging_sim = ChargingSimulator(
    vehicle_fleet=fleet,
    region=region,
    mobility_demand=mobility_sim,
    charging_efficiency=0.9,
    scenario={
        'home': {
            'share': 0.5,  # 50% of EVs charge at home
            'power_options_kW': [[3.7, 0.9], [7.4, 0.1]],    
            'arrival_time_h': [18, 2]  # Arrival time with mean and std deviation
        },
        'work': {
            'share': 0.3,  # 30% of EVs charge at work
            'power_options_kW': [[7.4, 0.9], [11, 0.1]],    
            'arrival_time_h': [9, 1]
        },
        'poi': {
            'share': 0.2,  # 20% of EVs charge at points of interest
            'power_options_kW': [[3.7, 0.1], [7.4, 0.3], [11, 0.6]]    
        }
    }
)

# Compute spatial and temporal charging demand
charging_sim.compute_spatial_demand()
charging_sim.compute_temporal_demand(time_step=0.1) # Time step in hours
# Other possible options (travel_time_home_work: float = 0.5, soc_threshold_mean: float = 0.6, soc_threshold_std_dev: float = 0.2)

# Optional: Apply smart charging to reduce peak demand
# charging_sim.apply_smart_charging(location=["home"], charging_strategy="peak_shaving", share=0.5)

# Save charging demand data and visualizations
charging_sim.to_csv("output/00_ChargingDemand.csv")
charging_sim.chargingdemand_total_to_map("output/00_ChargingDemand_total.html")
charging_sim.chargingdemand_pervehicle_to_map("output/00_ChargingDemand_pervehicle.html")
charging_sim.chargingdemand_nvehicles_to_map("output/00_ChargingDemand_n_vehicles.html")

# STEP 5: PV Simulation for calculating photovoltaic power production
# Initialize PVSimulator using location coordinates, module characteristics, and installation type

pv = PVSimulator(
    environment={
        'latitude': region.centroid_coords()[0],  
        'longitude': region.centroid_coords()[1],  
        'year': 2020  
    }, 
    pv_module={
        'efficiency': 0.22,
        'temperature_coefficient': -0.004  
    }, 
    installation={
        'type': 'rooftop',  # Options: 'rooftop' or 'groundmounted_fixed'
        'system_losses': 0.14
    }
)

pv.compute_pv_production()  # Calculate PV production based on the defined parameters
pv.results.to_csv("output/00_PVProduction.csv")  # Save PV production data

# STEP 6: EV-PV Synergy Analysis
# Calculate synergies between PV generation and EV charging demand over a defined time period

evpv = EVPVSynergies(pv=pv, ev=charging_sim, pv_capacity_MW=10)

# Calculate daily synergy metrics for the first week of January, adjusting recompute_probability as needed
synergy_metrics = evpv.daily_metrics("01-01", "01-07", recompute_probability=0.0)
synergy_metrics.to_csv("output/00_EVPVSynergies.csv") # Save synergy metrics data