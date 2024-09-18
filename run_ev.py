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

from evpv.mobilitysim import MobilitySim
from evpv.chargingscenario import ChargingScenario
from evpv.evcalculator import EVCalculator
from evpv import helpers as hlp

##############################
##### EV Charging Demand #####
##############################

# Create a new EV Calculator and compute both the mobility and charging demands in a given scenario
ev = EVCalculator(
    mobility_demand = {
        'target_area_geojson': 'input/gadm41_ETH_1_AddisAbeba.json', 
        'population_raster': 'input/GHS_POP_merged_4326_3ss_V1_0_R8andR9_C22_cropped.tif', 
        'destinations_csv': 'input/workplaces.csv', 
        'trips_per_inhabitant': 0.01, 
        'zone_width_km': 4,
        'ORS_key': None #'5b3ce3597851110001cf6248879c0a16f2754562898e0826e061a1a3'
    },
    ev_fleet = [[EVCalculator.preset_car, 1.0], [EVCalculator.preset_motorbike, 0.0]],
    charging_efficiency = 0.9,
    charging_curve_params = {
        "Origin": {
            "Share": 0.0, 
            "Arrival time": [18, 2], 
            "Smart charging": 1.0 
        },
        "Destination": {
            "Share": 1.0,
            "Arrival time": [9, 2],
            "Smart charging": 1.0 
        }}
    )

# Save the results
ev.save_results(output_folder = "output", prefix = "Scenario1")

