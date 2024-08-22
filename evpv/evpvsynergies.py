# coding: utf-8

""" 
EVPVSynergies

"""

import numpy as np
import pandas as pd
import geopandas as gpd
import json
import warnings
import os
import time
import math
import csv
from scipy.interpolate import interp1d
import scipy.integrate as integrate
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from pathlib import Path

from evpv import helpers as hlp

load_dotenv() # take environment variables from .env

INPUT_PATH = Path( str(os.getenv("INPUT_PATH")) )
OUTPUT_PATH = Path( str(os.getenv("OUTPUT_PATH")) )

class EVPVSynergies:

    #######################################
    ############# ATTRIBUTES ##############
    #######################################
    
    pv_capacity_factor = {}
    ev_charging_demand_MW = lambda x: None
    pv_capacity_MW = .0

    #######################################
    ############### METHODS ###############
    #######################################
    
    ############# Constructor #############
    ####################################### 

    def __init__(self, 
        pv_capacity_factor, 
        ev_charging_demand_MW,
        pv_capacity_MW):

        self.set_charging_demand_MW(ev_charging_demand_MW)
        self.set_pv_capacity_factor(pv_capacity_factor) 
        self.set_pv_capacity_MW(pv_capacity_MW)        
       

    ############# Setters #############
    ###################################

    def set_charging_demand_MW(self, ev_charging_demand_MW):
        # Extract the 'Time' and 'Total profile (MW)' columns
        time = ev_charging_demand_MW['Time']
        profile = ev_charging_demand_MW['Total profile (MW)']

        self.ev_charging_demand_MW = interp1d(time, profile, kind='linear', fill_value = 'extrapolate')

    def set_pv_capacity_factor(self, pv_capacity_factor):
        df = pv_capacity_factor

        # Rename the columns for convenience (optional, but helpful)
        df.columns = ['Timestamp', 'Total profile (MW)']

        # Convert the first column to datetime format
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])

        # Extract the 'Month-Day' and 'Hour' from the timestamp
        df['Month-Day'] = df['Timestamp'].dt.strftime('%m-%d')
        df['Hour'] = df['Timestamp'].dt.hour

        # Create a dictionary to hold the interpolation functions for each day
        interpolation_functions = {}

        # Group data by 'Day'
        grouped = df.groupby('Month-Day')

        # Create an interpolation function for each day
        for day, group in grouped:
            hours = group['Hour']
            profile = group['Total profile (MW)']
            
            # Create the interpolation function for this day
            interpolation_function = interp1d(hours, profile, kind='linear', fill_value = 'extrapolate')
            
            # Store the function in the dictionary with the day as the key
            interpolation_functions[day] = interpolation_function

        self.capacity_factor = interpolation_functions

    def set_pv_capacity_MW(self, pv_capacity_MW):
        self.pv_capacity_MW = pv_capacity_MW

    ########### PV Production #########
    ###################################

    def pv_power_MW(self, day='01-01'):
        return lambda x: self.capacity_factor[day](x-8) * self.pv_capacity_MW

    def pv_production(self, day='01-01'):        
        result, error = integrate.quad(self.pv_power_MW(day), 0, 24)
        return result

    ############# EV Demand ###########
    ###################################

    def ev_demand(self):        
        result, error = integrate.quad(self.ev_charging_demand_MW, 0, 24)
        return result

    ########## EV-PV Synergies ########
    ###################################

    def energy_coverage_ratio(self, day='01-01'):       
        return self.pv_production(day) / self.ev_demand()

    def self_sufficiency_ratio(self, day='01-01'):
        coincident_power = lambda x: min(self.pv_power_MW(day)(x), self.ev_charging_demand_MW(x))
        result, error = integrate.quad(coincident_power, 0, 24)

        return result / self.ev_demand()

    def self_consumption_ratio(self, day='01-01'):
        coincident_power = lambda x: min(self.pv_power_MW(day)(x), self.ev_charging_demand_MW(x))
        result, error = integrate.quad(coincident_power, 0, 24)

        return result / self.pv_production(day)

    def excess_pv_ratio(self, day='01-01'):
        coincident_power = lambda x: min(self.pv_power_MW(day)(x), self.ev_charging_demand_MW(x))
        result, error = integrate.quad(coincident_power, 0, 24)

        pv_prod = self.pv_production(day)

        return (pv_prod - result) / pv_prod

    def spearman_correlation(self, day='01-01', n_points = 100): 
        # Define the range and resolution
        t_values = np.linspace(0, 24, n_points) 

        pv_values = self.pv_power_MW(day)(t_values)
        ev_values = self.ev_charging_demand_MW(t_values)

        # Compute the Spearman rank correlation coefficient
        spearman_coef, p_value = spearmanr(pv_values, ev_values)

        return spearman_coef, p_value
    
        