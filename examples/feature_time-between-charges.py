# coding: utf-8

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import csv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Append parent directory to include evpv modules

from evpv import helpers as hlp

# Constants
battery_capacities = [60, 15]  # Two battery capacities: 60 kWh and 15 kWh
num_samples = 50000  # Number of samples to generate for the envelope
daily_demands = np.linspace(1, 8, 100)  # Daily energy demand ranging from 1 kWh to 8 kWh

# Store results
probabilities = {capacity: np.zeros_like(daily_demands) for capacity in battery_capacities}
std_devs = {capacity: np.zeros_like(daily_demands) for capacity in battery_capacities}

# Simulate the days between charges for each daily demand for both battery capacities
for battery_capacity in battery_capacities:
    for i, demand in enumerate(daily_demands):
        # Generate multiple samples to capture the variability
        samples = [hlp.calculate_days_between_charges_single_vehicle(demand, battery_capacity) for _ in range(num_samples)]
        
        # Calculate the mean and standard deviation
        mean_days = np.mean(samples)
        probabilities[battery_capacity][i] = 1 / mean_days if mean_days > 0 else 0  # Avoid division by zero
        std_dev_raw = np.std([1 / s for s in samples if s > 0])  # Standard deviation of probabilities
        std_devs[battery_capacity][i] = min(std_dev_raw, 1)  # Limit the standard deviation to a maximum of 1

# Save results to a CSV file
output_file = "results.csv"

with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header
    header = ["Daily Demand (kWh)", 
              "Probability (60 kWh)", "Std Dev (60 kWh)", 
              "Probability (15 kWh)", "Std Dev (15 kWh)"]
    writer.writerow(header)
    
    # Write the data
    for i in range(len(daily_demands)):
        row = [
            daily_demands[i],
            probabilities[60][i], std_devs[60][i],
            probabilities[15][i], std_devs[15][i]
        ]
        writer.writerow(row)

# Plotting the results
plt.figure(figsize=(10, 6))

# For each battery capacity, plot the probability and standard deviation envelope
for battery_capacity in battery_capacities:
    plt.plot(daily_demands, probabilities[battery_capacity], 
             label=f'Probability - {battery_capacity} kWh battery', linewidth=2)
    
    # Create the envelope using probability ± standard deviation
    plt.fill_between(daily_demands, 
                     np.maximum(probabilities[battery_capacity] - std_devs[battery_capacity], 0),  # Prevent negative bounds
                     np.minimum(probabilities[battery_capacity] + std_devs[battery_capacity], 1),  # Prevent bounds exceeding 1
                     alpha=0.3, label=f'Std Dev Envelope - {battery_capacity} kWh')

plt.title('Probability of Daily Energy Demand vs. Battery Capacities')
plt.xlabel('Daily Energy Demand (kWh)')
plt.ylabel('Probability (1/Mean Days)')
plt.legend()
plt.grid(True)
plt.show()
