# coding: utf-8

""" 
A script to calculate PV production in a given location using PVLib
"""

import pvlib
from pvlib.pvsystem import PVSystem, Array, FixedMount
from pvlib.location import Location
from pvlib.modelchain import ModelChain
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS

import pandas as pd
import matplotlib.pyplot as plt
import pytz

# Module 

sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
module = sandia_modules['Canadian_Solar_CS5P_220M___2009_']

# Inverter

sapm_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')
inverter = sapm_inverters['ABB__MICRO_0_25_I_OUTD_US_208__208V_']

# Thermal model to calculate the module/cell temperature

temperature_parameters = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']

# PV System and Installation conditions

system = PVSystem(surface_tilt = 0, surface_azimuth = 180,
	module_parameters = module, inverter_parameters = inverter,
	temperature_model_parameters = temperature_parameters,
	modules_per_string = 1, strings_per_inverter = 1)

# Location

location = Location(latitude = 9.005401, longitude = 38.763611, tz = 'Africa/Addis_Ababa', altitude = 2350, name = 'Addis_Ababa')

# Weather input data

# # Clear sky

# times = pd.date_range(start="2024-07-01", end="2024-07-07", freq="1min", tz = location.tz)
# clear_sky = location.get_clearsky(times) # Clear sky model

# # TMY data /!\ Still a problem with the time zone

# weather_data_tmy, months_selected, meta, inputs = pvlib.iotools.get_pvgis_tmy(latitude=location.latitude, longitude=location.longitude, 
# 	outputformat='json', usehorizon=True, userhorizon=None, 
# 	startyear=None, endyear=None, 
# 	map_variables=True, url='https://re.jrc.ec.europa.eu/api/v5_2/', timeout=30)

# weather_data_tmy.index = pd.date_range(start='2025-01-01 00:00', end='2025-12-31 23:00', freq='h')

# weather_data_tmy.index = weather_data_tmy.index.tz_localize('UTC')

# # Manually adjust for the timezone offset of '+03:00'
# local_time_offset = 3 

# offset = pd.Timedelta(hours = 3)  # Offset for '+03:00'
# weather_data_tmy.index = weather_data_tmy.index + offset

# weather_data_tmy.index = weather_data_tmy.index.tz_localize(None)

# # Move the last n rows to the beginning
# n = local_time_offset

# last_n_rows = weather_data_tmy.tail(n)
# remaining_rows = weather_data_tmy.head(len(weather_data_tmy) - n)
# weather_data_tmy = pd.concat([last_n_rows, remaining_rows])

# # Reattach the year information to the DataFrame
# weather_data_tmy.index = pd.date_range(start='2025-01-01', periods=len(weather_data_tmy), freq='h')

# print(weather_data_tmy)

# POA data for a given year PVGIS

weather_data_poa, meta, inputs = pvlib.iotools.get_pvgis_hourly(location.latitude, location.longitude, 
	start=2020, end=2020, 
	raddatabase='PVGIS-SARAH2', components=True, 
	surface_tilt=45, surface_azimuth=180, 
	outputformat='json', usehorizon=True, userhorizon=None, 
	pvcalculation=False, peakpower=None, pvtechchoice='crystSi', mountingplace='free', loss=0, trackingtype=0, 
	optimal_surface_tilt=True, optimalangles=False, 
	url='https://re.jrc.ec.europa.eu/api/v5_2/', map_variables=True, timeout=30)

weather_data_poa['poa_diffuse'] = weather_data_poa['poa_sky_diffuse'] + weather_data_poa['poa_ground_diffuse']
weather_data_poa['poa_global'] = weather_data_poa['poa_direct'] + weather_data_poa['poa_diffuse']

weather_data_poa.index = pd.to_datetime((weather_data_poa.index))

weather_data_poa= weather_data_poa.tz_convert('Africa/Addis_Ababa')

# # Move the last n rows to the beginning
n = 3

last_n_rows = weather_data_poa.tail(n)
remaining_rows = weather_data_poa.head(len(weather_data_poa) - n)
weather_data_poa = pd.concat([last_n_rows, remaining_rows])

# Reattach the year information to the DataFrame
weather_data_poa.index = pd.date_range(start='2025-01-01', periods=len(weather_data_poa), freq='h')

# Run

modelchain = ModelChain(system, location)

modelchain.run_model_from_poa(weather_data_poa) # POA
# modelchain.run_model(weather_data_tmy) # Clear sky or TMY weather data

# Visualise results

pv_prod = modelchain.results.ac
capacity_factor = pv_prod / 220

print(capacity_factor.mean())
capacity_factor.to_csv("pv_capacity_factor_AddisAbaba.csv", header=None)  # header=True includes the column name

pv_prod.plot(figsize=(16,9))
plt.show()