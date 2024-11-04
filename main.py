# coding: utf-8

from evpv.vehicle import Vehicle
from evpv.vehiclefleet import VehicleFleet
from evpv.region import Region
from evpv.mobilitysimulator import MobilitySimulator
from evpv.pvsimulator import PVSimulator
from evpv.chargingsimulator import ChargingSimulator
from evpv.evpvsynergies import EVPVSynergies

import numpy as np
import matplotlib.pyplot as plt

car = Vehicle(name = "car", battery_capacity_kWh = 50, consumption_kWh_per_km = 0.15)
motorcycle = Vehicle(name = "motorcycle", battery_capacity_kWh = 10, consumption_kWh_per_km = 0.01, max_charging_power_kW = 100)
fleet = VehicleFleet(total_vehicles = 1000, vehicle_types = [[car, 1.0]])

region = Region(
    region_geojson="examples/input/gadm41_ETH_1_AddisAbeba.json",    
    population_raster="examples/input/GHS_POP_merged_4326_3ss_V1_0_R8andR9_C22_cropped.tif",
    workplaces_csv="examples/input/workplaces.csv",
    pois_csv="examples/input/intermediate_stops.csv",
    traffic_zone_properties={
        "target_size_km": 5,
        "shape": "rectangle",
        "crop_to_region": True
    }
)

region.to_map("myregion.html")

mobility_sim = MobilitySimulator(
    vehicle_fleet=fleet,
    region=region,
    vehicle_allocation_params = {
        "method": "population",                
        "randomness": 0.0
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

mobility_sim.vehicle_allocation_to_map("mappp.html")
mobility_sim.trip_distribution_to_map("mappp.html", "2_2")


# ms = mobility_sim + mobility_sim2


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

# print(pv.results)

charging_sim = ChargingSimulator(
    vehicle_fleet=fleet,
    region=region,
    mobility_demand=mobility_sim,
    charging_efficiency=0.9,
    scenario = {
        'home': {
            'share': 0.5,  # 50% of EVs charge at home
            'power_options_kW': [ [3.7, 0.9], [7.4, 0.1]],    
            'arrival_time_h': [18, 2]
        },
        'work': {
            'share': 0.4,  # 30% of EVs charge at work
            'power_options_kW': [[7.4, 0.9], [11, 0.1]],    
            'arrival_time_h': [9, 1]
        },
        'poi': {
            'share': 0.1,  # 20% of EVs charge at pois
            'power_options_kW': [[3.7, 0.4], [22, 0.6]]    
        }
    }
)

charging_sim.compute_spatial_demand()
charging_sim.compute_temporal_demand(0.1)

charging_sim.chargingdemand_total_to_map("map1.html")
charging_sim.chargingdemand_pervehicle_to_map("map2.html")
charging_sim.chargingdemand_nvehicles_to_map("map3.html")



# print(charging_sim.temporal_demand_profile_aggregated)

evpv = EVPVSynergies(pv, charging_sim, 10)

# print(evpv.daily_metrics("01-01", "01-30", recompute_probability = 0.0))

# charging_sim.temporal_demand_profile_aggregated.to_csv("wo.csv")
# charging_sim.apply_smart_charging(location = ["home", "work", "poi"], share = 1.0, charging_strategy = "peak_shaving")
# charging_sim.temporal_demand_profile_aggregated.to_csv("w.csv")


