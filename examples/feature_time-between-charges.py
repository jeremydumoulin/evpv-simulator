# coding: utf-8

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Append parent directory to include evpv modules

from evpv import helpers as hlp

# Constants
battery_capacities = [60, 20]  # Two battery capacities: 50 kWh and 15 kWh
num_samples = 1000  # Number of samples to generate for the envelope
daily_demands = np.linspace(5, 30, 100)  # Daily energy demand ranging from 5 kWh to 30 kWh

# Store results
mean_days = {capacity: np.zeros_like(daily_demands) for capacity in battery_capacities}
percentile_20 = {capacity: np.zeros_like(daily_demands) for capacity in battery_capacities}
percentile_80 = {capacity: np.zeros_like(daily_demands) for capacity in battery_capacities}

# Simulate the days between charges for each daily demand for both battery capacities
for battery_capacity in battery_capacities:
    for i, demand in enumerate(daily_demands):
        # Generate multiple samples to capture the variability
        samples = [hlp.calculate_days_between_charges_single_vehicle(demand, battery_capacity) for _ in range(num_samples)]
        
        # Calculate the mean, 10th, and 90th percentiles
        mean_days[battery_capacity][i] = np.mean(samples)
        percentile_20[battery_capacity][i] = np.percentile(samples, 20)
        percentile_80[battery_capacity][i] = np.percentile(samples, 80)

# Plotting the results
plt.figure(figsize=(10, 6))

# For each battery capacity, plot the mean and percentile envelope
for battery_capacity in battery_capacities:
    plt.plot(daily_demands, mean_days[battery_capacity], label=f'Mean Days - {battery_capacity} kWh battery', linewidth=2)
    
    # Create the envelope using 20th and 80th percentiles
    plt.fill_between(daily_demands, 
                     percentile_20[battery_capacity], 
                     percentile_80[battery_capacity], 
                     alpha=0.3, label=f'20th-80th Percentile Envelope - {battery_capacity} kWh')

plt.title('Days Between Charges vs. Daily Energy Demand for Different Battery Capacities')
plt.xlabel('Daily Energy Demand (kWh)')
plt.ylabel('Days Between Charges')
plt.legend()
plt.grid(True)
plt.show()