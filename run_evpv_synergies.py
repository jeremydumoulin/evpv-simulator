# coding: utf-8

""" 
A p
"""

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import time
from datetime import datetime, timedelta
from scipy.interpolate import interp1d

from dotenv import load_dotenv
from pathlib import Path

from evpv.evpvsynergies import EVPVSynergies
from evpv import helpers as hlp

"""
Environment variables
"""

load_dotenv() # take environment variables from .env

INPUT_PATH = Path( str(os.getenv("INPUT_PATH")) )
OUTPUT_PATH = Path( str(os.getenv("OUTPUT_PATH")) )

"""
Data
"""

# PV capacity factor
pv_capacity_factor = pd.read_csv(INPUT_PATH/"pv_capacity_factor_AddisAbaba.csv")

# Daily charging curve
ev_power_profile = pd.read_csv(OUTPUT_PATH/"evpv_Result_ChargingDemand_PowerProfile.csv")
ev_power_profile = ev_power_profile[['Time', 'Total profile (MW)']] 

# Populate object for analysing EVPV Synergies
evpv_syn = EVPVSynergies(pv_capacity_factor = pv_capacity_factor, ev_charging_demand_MW = ev_power_profile, pv_capacity_MW = 1)

# User selects a specific day for the second interpolation

day = '01-11'  # MM-DD format, e.g., '01-01' for January 1st


# EV-PV KPIs

# Initialize a DataFrame to store the results
results_df = pd.DataFrame(columns=['Installed Capacity (MW)', 'PV Production (MWh)', 'EV Demand (MWh)', 'ECR (%)', 'SSR (%)', 'SCR (%)'])
capacities_to_investigate = range(10, 2001, 100)

# Loop through the parameters
for capacity in capacities_to_investigate:
    evpv_syn.set_pv_capacity_MW(capacity)
    print(capacity)

    # Calculate the indicators
    pv_prod = evpv_syn.pv_production(day)
    ev_demand = evpv_syn.ev_demand()
    ecr = evpv_syn.energy_coverage_ratio(day)
    ssr = evpv_syn.self_sufficiency_ratio(day)
    scr = evpv_syn.self_consumption_ratio(day)
    epr = evpv_syn.excess_pv_ratio(day)
    
    # Store the results in a dictionary
    result = pd.DataFrame({
        'Installed Capacity (MW)': [capacity],
        'PV Production (MWh)': [pv_prod],
        'EV Demand (MWh)': [ev_demand],
        'ECR (%)': [ecr * 100],
        'SSR (%)': [ssr * 100],
        'SCR (%)': [scr * 100],
        'EPR (%)': [epr * 100]
    })
    
    # Append the result to the DataFrame using pd.concat
    results_df = pd.concat([results_df, result], ignore_index=True)

# Save the DataFrame to a CSV file
results_df.to_csv('evpv_results.csv', index=False)


# Plot for a given capacity

pv_capacity_MW = 100
evpv_syn.set_pv_capacity_MW(pv_capacity_MW)
hour_range = np.linspace(0, 24, 1000)  # Hourly range from 0 to 23

# Generate data points for the first interpolation function
pv_production = evpv_syn.pv_power_MW(day)(hour_range)
ev_charging_demand = evpv_syn.ev_charging_demand_MW(hour_range)

cr = evpv_syn.spearman_correlation(day = day, n_points = 100)
print(cr)

# Plot the two interpolation functions
plt.figure(figsize=(12, 6))
plt.plot(hour_range, ev_charging_demand, label='EV Charging Demand', color='blue', linestyle='--')
plt.plot(hour_range, pv_production, label=f'PV Production - Installed capacity {pv_capacity_MW} MW', color='red', linestyle='-')
plt.xlabel('Time (h)')
plt.ylabel('Power (MW)')
plt.legend()
plt.grid(True)
plt.show()


