# coding: utf-8

""" 
ChargingDemand

A class to simulate the daily charging demand using mobility simulation and assumptions 
regarding the charging scenario and the EV fuel consumption and charging efficiency
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import json
import warnings
import os
import rasterio
from rasterio.mask import mask
from pathlib import Path
from shapely.geometry import shape, LineString, Point, Polygon, box, MultiPoint
from shapely.ops import transform, nearest_points, snap
import pyproj
from pyproj import Geod
from dotenv import load_dotenv
import math

from evpv import helpers as hlp
from evpv.mobilitysim import MobilitySim

load_dotenv() # take environment variables from .env

INPUT_PATH = Path( str(os.getenv("INPUT_PATH")) )
OUTPUT_PATH = Path( str(os.getenv("OUTPUT_PATH")) )

class ChargingDemand:

    #######################################
    ############# ATTRIBUTES ##############
    #######################################
    
    # Mobility Simulation
    mobsim = None

    # EV fuel consumption and charging efficiency
    ev_consumption = .0
    charging_efficiency = .0

    # Charging scenario definition
    charging_scenario = {}

    # Outputs
    taz_properties = pd.DataFrame() # Relevant properties from Mobility Simulation for each TAZ

    charging_curve = pd.DataFrame() # Charging curve for EVs 
    charging_distribution = pd.DataFrame() # Charging need distribution across TAZs

    #######################################
    ############### METHODS ###############
    #######################################
    
    ############# Constructor #############
    ####################################### 

    def __init__(self, mobsim, ev_consumption, charging_efficiency):
        print("---")
        print(f"INFO \t ChargingDemand object initialisation")

        self.set_mobsim(mobsim)
        self.set_charging_efficiency(charging_efficiency)
        self.set_ev_consumption(ev_consumption)

        self.set_taz_properties()

        print(f"INFO \t ChargingDemand object created")
        print(f" \t - Number of trips: {self.taz_properties['n_inflows'].sum()} (n_in) // {self.taz_properties['n_outflows'].sum()} (n_out)")
        print(f" \t - FKT (origin to destination): {self.taz_properties['fkt_inflows'].sum()} km")
        print(f" \t - Average VKT (origin to destination): {self.taz_properties['fkt_inflows'].sum() / self.taz_properties['n_inflows'].sum()} km")

        print("---")

    ############# Setters #############
    ###################################

    def set_mobsim(self, mobsim):
        """ Setter for the mobsim attribute.
        """
        if not isinstance(mobsim, MobilitySim):
            print(f"ERROR \t A MobilitySim object is required as an input.")
            return

        # Check if trip generation has been performed
        if not 'n_outflows' in mobsim.traffic_zones.columns:
            print(f"ERROR \t MobilitySim object - Trip Generation has not been performed.")
            return

        # Check if trip generation has been performed
        if not 'Origin' in mobsim.flows.columns:
            print(f"ERROR \t MobilitySim object - Trip Distribution has not been performed.")
            return

        self.mobsim = mobsim

    def set_ev_consumption(self, ev_consumption):
        self.ev_consumption = ev_consumption

    def set_charging_efficiency(self, charging_efficiency):
        self.charging_efficiency = charging_efficiency

    def set_taz_properties(self):
        """ Setter for the taz_properties attribute.
        """
        taz = self.mobsim.traffic_zones
        flows = self.mobsim.flows

        # Prepare lists to hold the data
        taz_id = []
        geometric_center = []
        nearest_node = []
        bbox = []
        n_outflows = []
        n_inflows = []
        fkt_outflows = []
        fkt_inflows = []
        vkt_inflows = []
        vkt_outflows = []

        # Iterate over the TAZ and append data
        for index, row in taz.iterrows():
            taz_id.append(row['id'])
            geometric_center.append(row['geometric_center'])
            nearest_node.append(row['nearest_node'])
            bbox.append(row['bbox'])

            # Append values related to the origin (outflows)            
            out_df = flows[flows['Origin'] == row['id']].copy()
            out_df['Distance_Flow_Product'] = out_df['Travel Distance (km)'] * out_df['Flow']

            n_outflows.append(out_df['Flow'].sum())
            fkt_outflows.append(out_df['Distance_Flow_Product'].sum()) 
            vkt_outflows.append(out_df['Distance_Flow_Product'].sum() / out_df['Flow'].sum())             

             # Append values related to the destination (inflows)            
            in_df = flows[flows['Destination'] == row['id']].copy()
            in_df['Distance_Flow_Product'] = in_df['Travel Distance (km)'] * in_df['Flow']

            n_inflows.append(in_df['Flow'].sum())
            fkt_inflows.append(in_df['Distance_Flow_Product'].sum()) 
            vkt_inflows.append(in_df['Distance_Flow_Product'].sum() / in_df['Flow'].sum()) 

            
        # Create the resulting DataFrame
        data = {
            'id': taz_id,
            'geometric_center': geometric_center,
            'nearest_node': nearest_node,
            'n_outflows': n_outflows,
            'n_inflows': n_inflows,
            'fkt_outflows': fkt_outflows,
            'fkt_inflows': fkt_inflows,
            'vkt_outflows': vkt_outflows,
            'vkt_inflows': vkt_inflows
        }

        taz_properties= pd.DataFrame(data)    

        self.taz_properties = taz_properties