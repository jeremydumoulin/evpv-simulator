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

from evpv.pvcalculator import PVCalculator

"""
Environment variables
"""

load_dotenv() # take environment variables from .env

INPUT_PATH = Path( str(os.getenv("INPUT_PATH")) )
OUTPUT_PATH = Path( str(os.getenv("OUTPUT_PATH")) )

"""
PV Scenario
"""

pv_calc = PVCalculator(
    environment = {
        'latitude': 9.005401,
        'longitude': 38.763611,
        'year': 2020
        }, 
    pv_module = {
        'efficiency': 0.22,
        'temperature_coefficient':-0.0035
        }, 
    installation = {
        'tracking': 'fixed',
        'tilt': None,
        'azimuth': None,
    })


pv_prod = pv_calc.compute_pv_production()


# Assuming your DataFrame is named 'df'
pv_prod['PV Production (W/m2)'].plot(figsize=(10, 6))

plt.title('POA Global Over Time')
plt.xlabel('Time')
plt.ylabel('POA Global (W/m^2)')
plt.grid(True)

# Display the plot
plt.show()