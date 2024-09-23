# coding: utf-8

""" 
A python script illustrating the basic usage of the EV-PV model
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Append parent directory to include evpv

from evpv.evcalculator import EVCalculator
from evpv.pvcalculator import PVCalculator
from evpv.evpvsynergies import EVPVSynergies

######################################
###### Scenario Name (Optional) ######
######################################

sc = 'S1'

######################################
##### STEP 1: EV Charging Demand #####
######################################

# EV Calculator object: computes the mobility demand and charging demand for a given set of parameters
ev = EVCalculator(
    mobility_demand = {
        # Required parameters
        'target_area_geojson': 'input/gadm41_ETH_1_AddisAbeba.json', 
        'population_raster': 'input/GHS_POP_merged_4326_3ss_V1_0_R8andR9_C22_cropped.tif', 
        'destinations_csv': 'input/workplaces.csv', 
        'trips_per_inhabitant': 0.1, 
        'zone_width_km': 4,
        # Optional
        'ORS_key': '5b3ce3597851110001cf6248879c0a16f2754562898e0826e061a1a3'
    },
    ev_fleet = [
        [EVCalculator.preset['car'], 1.0], 
        [EVCalculator.preset['motorbike'], 0.0]
    ],
    charging_efficiency = 0.9,
    charging_scenario = {
        "Home": {
            "Share": 0.0, 
            "Arrival time": [18, 2], 
            "Smart charging": 0.0 
        },
        "Destination": {
            "Share": 1.0,
            "Arrival time": [9, 2],
            "Smart charging": 1.0 
        }}
    )

# Save the results
ev.save_results(output_folder = "output", prefix = sc)

######################################
######## STEP 2: PV Production #######
######################################

# PV Calculator object: computes PV KPIs for a given set of parameters
pv = PVCalculator(
    environment = {
        'latitude': 9.005401,
        'longitude': 38.763611,
        'year': '2020'
        }, 
    pv_module = {
        'efficiency': 0.22,
        'temperature_coefficient': -0.0035
        }, 
    installation = {
        'type': 'groundmounted_fixed'
    })

# Run the PV simulation
pv_prod = pv.compute_pv_production()

# Save the results
pv_prod.to_csv(f"output/{sc}_PVproduction.csv")

######################################
####### STEP 3: EV-PV Synergies ######
######################################

# Inputs from previous results
capacity_factor = pv_prod['Capacity Factor'].reset_index() 
charging_curve = ev.charging_demand.charging_profile[['Time', 'Total (MW)']] 

# EVPVSynergies object: computes EV-PV KPIs
evpv = EVPVSynergies(
    pv_capacity_factor = capacity_factor, 
    ev_charging_demand_MW = charging_curve, 
    pv_capacity_MW = 280)

# One example KPI for a single day
print(evpv.self_sufficiency_ratio(day='01-01'))

# All KPIs over a given period
daily_kpis = evpv.daily_metrics(start_date = '01-01', end_date = '01-30')
daily_kpis.to_csv(f"output/{sc}_EVPV_KPIs.csv")

