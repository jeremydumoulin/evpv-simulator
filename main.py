# coding: utf-8

from evpv.vehicle import Vehicle
from evpv.vehiclefleet import VehicleFleet
from evpv.region import Region
from evpv.mobilitysimulator import MobilitySimulator
from evpv.pvsimulator import PVSimulator
from evpv.chargingsimulator import ChargingSimulator

import numpy as np
import matplotlib.pyplot as plt

car = Vehicle(name = "car", battery_capacity_kwh = 70, consumption_kwh_per_km = 0.2)
motorcycle = Vehicle(name = "motorcycle", battery_capacity_kwh = 10, consumption_kwh_per_km = 0.06)
fleet = VehicleFleet(total_vehicles = 10000, vehicle_types = [[car, 0.5], [motorcycle, 0.5]])

region = Region(
    region_geojson="examples/input/gadm41_ETH_1_AddisAbeba.json",    
    population_raster="examples/input/GHS_POP_merged_4326_3ss_V1_0_R8andR9_C22_cropped.tif",
    workplaces_csv="examples/input/workplaces.csv",
    pois_csv="examples/input/intermediate_stops.csv",
    traffic_zone_properties={
        "target_size_km": 4,
        "shape": "rectangle",
        "crop_to_region": True
    }
)

mobility_sim = MobilitySimulator(
    vehicle_fleet=fleet,
    region=region,
    vehicle_allocation_params = {
        "method": "population",                
        "randomness": 0.7
    }, 
    trip_distribution_params = {
        "model_type": "gravity_exp_scaled",                
        "attraction_feature": "workplaces",     
        "cost_feature": "distance_road",             
        "road_to_euclidian_ratio": 1.63,         
        "ors_key": None, # "5b3ce3597851110001cf6248879c0a16f2754562898e0826e061a1a3" 
        "distance_offset_km": 0.0                         
    }
)
mobility_sim.vehicle_allocation() 
mobility_sim.trip_distribution()

mobility_sim.to_csv("mobility_sim.csv")

# pv = PVSimulator(
#     environment = {
#         'latitude': region.centroid_coords()[0],  
#         'longitude': region.centroid_coords()[1],  
#         'year': 2020  
#         }, 
#     pv_module = {
#         'efficiency': 0.22,
#         'temperature_coefficient': -0.004  
#         }, 
#     installation = {
#         'type': 'rooftop',  # groundmounted_fixed
#         'system_losses': 0.14
#     })
# pv.compute_pv_production()

charging_sim = ChargingSimulator(
    vehicle_fleet=fleet,
    region=region,
    mobility_demand=mobility_sim,
    scenario = {
        'home': {
            'share': 0.5,  # 50% of EVs charge at home
            'power_options': [[3.7, 0.6], [7.4, 0.3], [11, 0.1]],    
            'arrival_time': [9, 2]
        },
        'work': {
            'share': 0.3,  # 50% of EVs charge at home
            'power_options': [[3.7, 0.6], [7.4, 0.3], [11, 0.1]],    
            'arrival_time': [9, 2]
        },
        'pois': {
            'share': 0.2,  # 50% of EVs charge at home
            'power_options': [[3.7, 0.6], [7.4, 0.3], [11, 0.1]],    
            'arrival_time': [9, 2]
        }
    }
)