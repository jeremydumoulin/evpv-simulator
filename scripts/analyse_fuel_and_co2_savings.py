# coding: utf-8

"""
A script to assess the fuel cost and CO₂ emission savings from switching from ICEVs to EVs,  
based on simulated vehicle flow and travel distances of the evpv model.

## Description:
This script processes the output of the mobility simulation to estimate the environmental  
and financial benefits of EV adoption. It calculates:  
1. **CO₂ emission savings per vehicle** by comparing ICEV and EV emissions.  
2. **Fuel cost savings per vehicle** by comparing ICEV fuel costs and EV charging costs.  
3. **Total and per-vehicle savings**, weighted by vehicle flow.  
4. **Histograms of CO₂ and fuel cost savings**, weighted by vehicle flow.  
5. **Summary results saved to a CSV file** for further analysis.  

## Usage:
1. Update the `mobility_sim_res` variable with the correct path to the simulation results file.  
2. Adjust key parameters such as:  
   - `carbon_intensity_elec`: Carbon intensity of the electricity mix (kgCO₂/kWh).  
   - `fuel_consumption_per_km`: ICEV fuel consumption (L/km).  
3. Run the script. The results will be visualized as histograms and saved in `co2_fuel_savings_results.csv`.  

## Dependencies:
- **pandas**: For reading the input CSV and performing calculations.  
- **numpy**: For numerical operations and binning histogram data.  
- **matplotlib**: To generate histograms of CO₂ and fuel cost savings.  

"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Parameters - PLEASE CHANGE

mobility_sim_res = "C:/Users/dumoulin/Documents/_CODES/evpv_addis_ababa/output/MobilitySimulation_results_flows.csv" 

nbr_days = 365 # Number of days to simulate

carbon_intensity_elec = 0.05  # kgCO2/kWh
ev_kwh_per_km = 0.183  # kWh/km
charging_efficiency = 0.9  # 90% efficient
icev_emission_per_km = 0.165 # kgCO2/km

fuel_price_per_liter = 1.0  # Example fuel price ($/liter)
fuel_consumption_per_km = 0.08  # ICEV fuel consumption (liters/km)
electricity_price_per_kwh = 0.04  # Example electricity price ($/kWh)

# Read input data

df = pd.read_csv(mobility_sim_res)

# Functions

def calculate_co2_savings(df, carbon_intensity_elec, ev_kwh_per_km, charging_efficiency, icev_emission_per_km):
    """
    Calculate CO2 savings by switching from ICEVs to EVs.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'Flow', 'Travel Distance (km)' columns.
    carbon_intensity_elec (float): Carbon intensity of electricity (kgCO2/kWh).
    ev_kwh_per_km (float): Energy consumption of EV per km.
    charging_efficiency (float): Efficiency of EV charging (e.g., 0.9 for 90% efficiency).
    icev_emission_per_km (float): Direct ICEV emissions (kgCO2/km).

    Returns:
    pd.Series: CO2 savings per OD pair.
    """

    # EV emissions per km
    ev_emission_per_km = (carbon_intensity_elec * ev_kwh_per_km) / charging_efficiency

    # Calculate emissions for ICEVs and EVs
    df["ICEV Emissions (kgCO2)"] = df["Travel Distance (km)"] * icev_emission_per_km
    df["EV Emissions (kgCO2)"] = df["Travel Distance (km)"] * ev_emission_per_km

    # Calculate CO2 savings
    df["CO2 Savings per EV (kgCO2)"] = nbr_days * 2 * (df["ICEV Emissions (kgCO2)"] - df["EV Emissions (kgCO2)"]) # Multiply by 2 to account for round trips

    return df["CO2 Savings per EV (kgCO2)"]

def calculate_fuel_cost_savings(df, fuel_price_per_liter, fuel_consumption_per_km, electricity_price_per_kwh, ev_kwh_per_km, charging_efficiency):
    """
    Calculate fuel cost savings from switching from ICEVs to EVs.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'Flow', 'Travel Distance (km)' columns.
    fuel_price_per_liter (float): Cost of fuel per liter ($/L).
    fuel_consumption_per_km (float): Fuel consumption of ICEVs (L/km).
    electricity_price_per_kwh (float): Cost of electricity per kWh ($/kWh).
    ev_kwh_per_km (float): Energy consumption of EVs per km.
    charging_efficiency (float): Efficiency of EV charging (e.g., 0.9 for 90% efficiency).

    Returns:
    pd.Series: Fuel cost savings per OD pair.
    """

    # Cost per km for ICEVs
    icev_cost_per_km = fuel_price_per_liter * fuel_consumption_per_km

    # Cost per km for EVs (including charging efficiency)
    ev_cost_per_km = (electricity_price_per_kwh * ev_kwh_per_km) / charging_efficiency

    # Calculate daily fuel cost savings per vehicle
    df["Fuel Cost Savings per EV ($)"] = nbr_days * 2 * (df["Travel Distance (km)"] * (icev_cost_per_km - ev_cost_per_km))

    return df["Fuel Cost Savings per EV ($)"]

# Run

co2_savings = calculate_co2_savings(df, carbon_intensity_elec, ev_kwh_per_km, charging_efficiency, icev_emission_per_km)
fuel_cost_savings = calculate_fuel_cost_savings(df, fuel_price_per_liter, fuel_consumption_per_km, electricity_price_per_kwh, ev_kwh_per_km, charging_efficiency)

# Compute total CO2 and fuel cost savings
total_co2_savings = (df["CO2 Savings per EV (kgCO2)"] * df["Flow"]).sum()
total_fuel_cost_savings = (df["Fuel Cost Savings per EV ($)"] * df["Flow"]).sum()

# Print total savings
print(f"Total CO2 Savings: {total_co2_savings:.2f} kgCO2")
print(f"Total Fuel Cost Savings: ${total_fuel_cost_savings:.2f}")

# Compute **weighted** average CO2 savings per vehicle
total_flow = df["Flow"].sum()

weighted_avg_savings_per_vehicle = (df["CO2 Savings per EV (kgCO2)"] * df["Flow"]).sum() / total_flow

# Compute **weighted** average fuel cost savings per vehicle
weighted_avg_fuel_cost_savings = (df["Fuel Cost Savings per EV ($)"] * df["Flow"]).sum() / total_flow

# Plot **weighted** histogram
plt.figure(figsize=(8, 5))
hist_bins = np.linspace(df["CO2 Savings per EV (kgCO2)"].min(), df["CO2 Savings per EV (kgCO2)"].max(), 20)
plt.hist(df["CO2 Savings per EV (kgCO2)"], bins=hist_bins, weights=df["Flow"], color='blue', alpha=0.7, edgecolor='black')

# Add weighted average as a vertical line
plt.axvline(weighted_avg_savings_per_vehicle, color='red', linestyle='dashed', linewidth=2, label=f'Weighted Avg: {weighted_avg_savings_per_vehicle:.2f} kg')

plt.xlabel("CO2 Savings per EV (kg)")
plt.ylabel("Number of EVs (Weighted by Flow)")
plt.title("CO2 Emission Savings per Vehicle from EV Adoption")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# Plot **weighted** histogram for Fuel Cost Savings
plt.figure(figsize=(8, 5))
hist_bins = np.linspace(df["Fuel Cost Savings per EV ($)"].min(), df["Fuel Cost Savings per EV ($)"].max(), 20)
plt.hist(df["Fuel Cost Savings per EV ($)"], bins=hist_bins, weights=df["Flow"], color='green', alpha=0.7, edgecolor='black')

# Add weighted average as a vertical line
plt.axvline(weighted_avg_fuel_cost_savings, color='red', linestyle='dashed', linewidth=2, label=f'Weighted Avg: ${weighted_avg_fuel_cost_savings:.2f}')

plt.xlabel("Fuel Cost Savings per EV ($)")
plt.ylabel("Number of EVs (Weighted by Flow)")
plt.title("Fuel Cost Savings per Vehicle from EV Adoption")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()


# Save results to CSV
output_csv = "co2_fuel_savings_results.csv"
summary_data = {
    "Metric": [
        "Weighted Avg CO2 Savings per EV (kgCO2)",
        "Weighted Avg Fuel Cost Savings per EV ($)",
        "Total CO2 Savings (kgCO2)",
        "Total Fuel Cost Savings ($)"
    ],
    "Value": [
        weighted_avg_savings_per_vehicle,
        weighted_avg_fuel_cost_savings,
        total_co2_savings,
        total_fuel_cost_savings
    ]
}

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(output_csv, index=False)

print(f"Summary results saved to {output_csv}")
