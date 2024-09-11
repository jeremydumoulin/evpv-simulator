# coding: utf-8

""" 
A python script to simulate the spatio-temporal charging demand based on 
mobility simulation.
"""

import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import LineString, Point, box, Polygon
from shapely import wkt
from shapely.ops import transform
from rasterio.features import geometry_mask
from rasterio.features import shapes
from shapely.geometry import mapping
import pyproj
from pyproj import Transformer
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from branca.colormap import LinearColormap
import numpy as np
import math
import os
import folium
from folium.plugins import AntPath
from folium.plugins import HeatMap
import pickle

from dotenv import load_dotenv
from pathlib import Path
import osmnx as ox
import branca.colormap as cm

from evpv.mobilitysim import MobilitySim
from evpv.chargingscenario import ChargingScenario
from evpv import helpers as hlp

#############################################
# PARAMETERS - MODIFY ACCORDING TO YOUR NEEDS
#############################################

"""
Environment variables
"""

load_dotenv() # take environment variables from .env

INPUT_PATH = Path( str(os.getenv("INPUT_PATH")) )
OUTPUT_PATH = Path( str(os.getenv("OUTPUT_PATH")) )
ORS_KEY = os.getenv("ORS_KEY")

"""
Global parameters 
"""

shapefile_path = INPUT_PATH / "gadm41_ETH_1_AddisAbeba.json" # Addis Ababa administrative boundaries
population_density_path = INPUT_PATH / "GHS_POP_merged_4326_3ss_V1_0_R8andR9_C22_cropped.tif" # Population density raster
destinations_path = INPUT_PATH /  "workplaces.csv"

taz_target_width_km = 3 # Desired TAZ width
simulation_area_extension_km = 0
population_to_ignore = 0.00

share_active = 0.626
share_unemployed = 0.23
share_home_office = 0.0
mode_share = 1.0
vehicle_occupancy = 1.2
ev_rate = 1.0 

n_trips_per_inhabitant = (share_active * (1 - share_unemployed) * (1 - share_home_office)) *  (mode_share / vehicle_occupancy) * ev_rate

vkt_offset = 0
model = "gravity_exp_02"
attraction_feature = "destinations"
cost_feature = "distance_road"

use_cached_data = True

#############################################
## MOBILITY SIMULATION 1 (home-work-home) ###
#############################################

# MobilitySim 

if os.path.isfile(OUTPUT_PATH / "evpv_Tmp_MobilitySim_Cache.pkl") and use_cached_data:
    mobsim = MobilitySim.from_pickle(OUTPUT_PATH / "evpv_Tmp_MobilitySim_Cache.pkl")

else:
    mobsim = MobilitySim(
        target_area = shapefile_path,
        population_density = population_density_path, 
        destinations = destinations_path)

    mobsim.setup_simulation(taz_target_width_km = taz_target_width_km, simulation_area_extension_km = simulation_area_extension_km, population_to_ignore_share = population_to_ignore)
    mobsim.trip_generation(n_trips_per_inhabitant = n_trips_per_inhabitant)     
    mobsim.trip_distribution(model = model, ors_key = ORS_KEY, attraction_feature = attraction_feature, cost_feature = cost_feature, vkt_offset = vkt_offset)

    mobsim.to_pickle(OUTPUT_PATH / f"evpv_Tmp_MobilitySim_Cache.pkl")

# Printing FKT and VKT
# print(f"FKT = ({mobsim.fkt} +/- {mobsim.fkt_error}) km")
# print(f"VKT = ({mobsim.vkt} +/- {mobsim.vkt_error}) km")

# Storing outputs

# All flows and TAZ properties

mobsim.flows.to_csv(OUTPUT_PATH / "evpv_Result_MobilitySim_OriginDestinationFlows.csv", index=False) # Store aggregated TAZ features as csv
mobsim.traffic_zones.to_csv(OUTPUT_PATH / "evpv_Result_MobilitySim_TrafficAnalysisZones.csv", index=False) # Store aggregated TAZ features as csv

# Histogram of VKTs 

vkt_distribution = mobsim.vkt_histogram(n_bins = 200)
vkt_distribution.to_csv(OUTPUT_PATH / "evpv_Result_MobilitySim_VKThistogram.csv", index=False)

# Maps
# mobsim.setup_to_map().save(OUTPUT_PATH / "evpv_Result_MobilitySim_SimulationSetup.html")
mobsim.trip_generation_to_map().save(OUTPUT_PATH / "evpv_Result_MobilitySim_TripGeneration.html")
mobsim.trip_distribution_to_map(trip_id = "6_1").save(OUTPUT_PATH / "evpv_Result_MobilitySim_TripDistribution.html")

#############################################
############### CHARGING NEEDS ##############
#############################################

cs = ChargingScenario(
    mobsim = [mobsim],
    ev_consumption = 0.2,
    charging_efficiency = 0.9,
    time_step = 1/10,
    scenario_definition = {
    "Origin": {
        "Share": 0.0, # Charging location share
        "Charging power": [[11, 1.0]], # Charging powers and shares of each charger
        "Arrival time": [18, 2], # Average charger plugin time and std deviation
        "Smart charging": .0 # Share of smart charging
    },
    "Destination": {
        "Share": 1.0,
        "Charging power": [[11, 1.0]], 
        "Arrival time": [9, 2],
        "Smart charging": 0.1 
    }
})

# Store spatial and temporal results as CSV
cs.charging_demand.to_csv(OUTPUT_PATH / "evpv_Result_ChargingDemand_Destination.csv", index=False) 
cs.charging_profile.to_csv(OUTPUT_PATH / "evpv_Result_ChargingDemand_PowerProfile.csv", index=False)

# Maps
cs.chargingdemand_total_to_map().save(OUTPUT_PATH / "evpv_Result_ChargingScenario_TotalChargingDemand_Destination.html")
cs.chargingdemand_pervehicle_to_map().save(OUTPUT_PATH / "evpv_Result_ChargingScenario_ChargingDemandPerCar_Destination.html")
cs.chargingdemand_nvehicles_to_map().save(OUTPUT_PATH / "evpv_Result_ChargingScenario_NumberOfVehicles_Destination.html")
