# coding: utf-8

from evpv.vehicle import Vehicle
from evpv.vehiclefleet import VehicleFleet
from evpv.region import Region
from evpv.mobilitysimulator import MobilitySimulator
from evpv.pvsimulator import PVSimulator

import numpy as np
import matplotlib.pyplot as plt

# car = Vehicle(name = "car", battery_capacity_kwh = 70, consumption_kwh_per_km = 0.2)
# motorcycle = Vehicle(name = "motorcycle", battery_capacity_kwh = 10, consumption_kwh_per_km = 0.06)
# fleet = VehicleFleet(total_vehicles = 10000, vehicle_types = [[car, 0.5], [motorcycle, 0.5]])

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

# mobility_sim = MobilitySimulator(
#     vehicle_fleet=fleet,
#     region=region,
#     vehicle_allocation_params = {
#         "method": "population",                
#         "randomness": 0.7
#     }, 
#     trip_distribution_params = {
#         "model_type": "gravity_exp_scaled",                
#         "attraction_feature": "workplaces",     
#         "cost_feature": "distance_road",             
#         "road_to_euclidian_ratio": 1.63,         
#         "ors_key": None, # "5b3ce3597851110001cf6248879c0a16f2754562898e0826e061a1a3" 
#         "distance_offset_km": 0.0                         
#     }
# )
# mobility_sim.vehicle_allocation() 
# mobility_sim.trip_distribution()

pv = PVSimulator(
    environment = {
        'latitude': 48.864716, #region.centroid_coords()[0],  
        'longitude': 2.349014, #region.centroid_coords()[1],  
        'year': '2020'  
        }, 
    pv_module = {
        'efficiency': 0.22  
        }, 
    installation = {
        'type': 'rooftop'  # groundmounted_fixed
    })
pv.compute_pv_production()

