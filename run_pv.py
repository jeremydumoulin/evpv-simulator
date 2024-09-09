# coding: utf-8

""" 
A p
"""

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as colors
import numpy as np
import math
import os
import time
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
import seaborn as sns

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

print(pv_prod)


# Assuming your DataFrame is named 'df'
# pv_prod['Performance Ratio'].plot(figsize=(10, 6))

# plt.title('POA Global Over Time')
# plt.xlabel('Time')
# plt.ylabel('POA Global (W/m^2)')
# plt.grid(True)

# # Display the plot
# plt.show()

# Heatmap Capacity factor

df = pv_prod

# Extract hour and day of year
df['hour'] = df.index.hour
df['date'] = df.index.date

# Create a pivot table for the heatmap
pivot_data = df.pivot(index='date', columns='hour', values='Capacity Factor')

# Create a custom colormap
viridis = plt.cm.get_cmap('viridis')
newcolors = viridis(np.linspace(0, 1, 256))
white = np.array([0.9, 0.9, 0.9, 1])
newcolors[:1, :] = white
newcmp = colors.ListedColormap(newcolors)

# Create the heatmap
plt.figure(figsize=(20, 10))
ax = sns.heatmap(pivot_data, cmap=newcmp,  
                 cbar_kws={'label': 'Capacity Factor'})

plt.title("Capacity Factor for PV Installation", fontsize=16)
plt.xlabel("Hour of the Day", fontsize=12)
plt.ylabel("Date", fontsize=12)

# Adjust x-axis labels to show all hours
ax.set_xticks(range(24))
ax.set_xticklabels(range(24))
plt.xticks(rotation=0)

# Adjust y-axis labels to show months
ax.yaxis.set_major_locator(mdates.MonthLocator())
ax.yaxis.set_major_formatter(mdates.DateFormatter('%b'))

# Improve tick label visibility
ax.tick_params(axis='both', which='major', labelsize=10)

plt.tight_layout()
plt.show()