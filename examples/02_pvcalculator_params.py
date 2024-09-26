# coding: utf-8

""" 
A python script illustrating the usage of the PVCalculator class, notably showing all required/optional parameters 
All the optional parameters are populated with the default inputs
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Append parent directory to include evpv

from evpv.pvcalculator import PVCalculator

######################################
## Prefix for results (for clarity) ##
######################################

pr = '02'

######################################
########## PV Calculator #############
######################################

pv = PVCalculator(
    environment = {
        # REQUIRED
        'latitude': 9.005401, # Latitude
        'longitude': 38.763611, # Longitude
        'year': '2020', # Reference year
        }, 
    pv_module = {
        # REQUIRED
        'efficiency': 0.22, # Efficiency in standard test conditions (STC)
        # OPTIONAL
        'temperature_coefficient': -0.0035 # Relative efficiency loss per K
        }, 
    installation = {
        # REQUIRED
        'type': 'groundmounted_fixed', # Type of PV system. Available options: rooftop, groundmounted_fixed, groundmounted_dualaxis, groundmounted_singleaxis_horizontal, groundmounted_singleaxis_vertical
        # OPTIONAL
        'system_losses': 0.14 # System losses (14% by default)
    })

# Run the PV simulation
pv.compute_pv_production()

# Save the results
pv.results.to_csv(f"output/{pr}_PVproduction.csv")
