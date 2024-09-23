# coding: utf-8

import pvlib
from pvlib import location, pvsystem, modelchain

import pandas as pd
from timezonefinder import TimezoneFinder
import pytz
from datetime import datetime

class PVCalculator:
    #######################################
    ############# Constructor #############
    #######################################

    def __init__(self, environment, pv_module, installation):
        print("")
        print(f"INFO \t Creating a new PV Calculator object")

        # Initialize the environment attributes
        self._environment = {
            # REQUIRED
            'latitude': environment.get('latitude'),
            'longitude': environment.get('longitude'),
            'year': environment.get('year'),
            # OPTIONAL
            'timezone': environment.get('timezone', self.get_timezone(environment.get('latitude'), environment.get('longitude')))
        }

        # Initialize the installation attributes
        self._installation = {
            'type': installation.get('type', 'groundmounted_fixed'), # groundmounted_fixed, groundmounted_dualaxis, groundmounted_singleaxis_horizontal, groundmounted_singleaxis_vertical
            'system_losses': installation.get('system_losses', 0.14)
        }

        # Initialize the PV device attributes
        self._pv_module = {
            'efficiency': pv_module.get('efficiency'),
            'temperature_coefficient': pv_module.get('temperature_coefficient', -0.0035)
        }

        # Create location, weather data and PV system objects
        self.location = self._create_location()
        self.weather_data =  self._fetch_weather_data()
        self.pv_system = self._create_pv_system()

    #######################################
    ### Parameters Setters and Getters ####
    #######################################

    @property
    def environment(self):
        return self._environment

    @environment.setter
    def environment(self, value):
        if isinstance(value, dict):
            self._environment.update(value)
        else:
            raise ValueError("Environment must be a dictionary")

    @property
    def installation(self):
        return self._installation

    @installation.setter
    def installation(self, value):
        if isinstance(value, dict):
            self._installation.update(value)
        else:
            raise ValueError("Installation must be a dictionary")

    @property
    def pv_module(self):
        return self._pv_module

    @pv_module.setter
    def pv_module(self, value):
        if isinstance(value, dict):
            self._pv_module.update(value)
        else:
            raise ValueError("PV module must be a dictionary")

    #######################################
    #### Location, Weather, PV System #####
    #######################################

    def _create_location(self):
        print(f"INFO \t Creating location object for timezone {self.environment['timezone']}")

        return location.Location(
            latitude=self.environment['latitude'],
            longitude=self.environment['longitude'],
            tz=self.environment['timezone']
        )

    def _fetch_weather_data(self):
        """ Get weather data from PVGIS in Africa with POA irradiance """

        print(f"INFO \t Fetching hourly weather data with POA irrdiance from PV GIS for the year {self.environment['year']} - Installation type: {self.installation['type']}")

        # Initialize tilt and azimuth
        tilt = 0 # Default value
        azimuth = 180 # Default value 
        optimize_tilt = optimize_azimuth = True

        # Set the tracking and tilt/azimuth options
        if self.installation['type'] == 'groundmounted_fixed':
            trackingtype = 0
        elif self.installation['type'] == 'groundmounted_singleaxis_horizontal':
            trackingtype = 1
        elif self.installation['type'] == 'groundmounted_singleaxis_vertical':
            trackingtype = 3
        elif self.installation['type'] == 'groundmounted_dualaxis':
            trackingtype = 2
        elif self.installation['type'] == 'rooftop':
            trackingtype = 0
            optimize_tilt = optimize_azimuth = False                 
            azimuth = 180
            tilt = 0
        else:
            raise ValueError(f"PV installation type is unknown")

        # Get data from PVGIS
        weather_data_poa, meta, inputs = pvlib.iotools.get_pvgis_hourly(self.location.latitude, self.location.longitude, 
            start=self.environment['year'], end=self.environment['year'], 
            raddatabase='PVGIS-SARAH2', components=True, 
            surface_tilt=tilt, surface_azimuth=azimuth, 
            outputformat='json', usehorizon=True, userhorizon=None, 
            pvcalculation=False, peakpower=None, pvtechchoice='crystSi', mountingplace='free', loss=0, 
            trackingtype=trackingtype, optimal_surface_tilt=optimize_tilt, optimalangles=optimize_azimuth, 
            url='https://re.jrc.ec.europa.eu/api/v5_2/', map_variables=True, timeout=30)

        # Get Diffuse and Global Irradiance in POA
        weather_data_poa['poa_diffuse'] = weather_data_poa['poa_sky_diffuse'] + weather_data_poa['poa_ground_diffuse']
        weather_data_poa['poa_global'] = weather_data_poa['poa_direct'] + weather_data_poa['poa_diffuse']

        # Convert the index to datetime
        weather_data_poa.index = pd.to_datetime( weather_data_poa.index)
        weather_data_poa.index = pd.to_datetime((weather_data_poa.index))

        # Convert the to local timezone
        weather_data_poa = weather_data_poa.tz_convert(self.environment['timezone'])

        # Because of the converting of the time zone, the last rows could be those of the next year
        # Here, we detect how many rows we have and shift them to the beginning of the data
        tz = pytz.timezone(self.environment['timezone']) 
        n = int( tz.localize(datetime.utcnow()).utcoffset().total_seconds() / 3600 ) # Get the number of hours from UTC

        last_n_rows = weather_data_poa.tail(n)
        remaining_rows = weather_data_poa.head(len(weather_data_poa) - n)
        weather_data_poa = pd.concat([last_n_rows, remaining_rows])

        # Reattach the year information to the DataFrame
        weather_data_poa.index = pd.date_range(start=f'{self.environment['year']}-01-01', periods=len(weather_data_poa), freq='h')

        # Print some information
        print(f"INFO \t > Elevation used for calculation: {meta['location']['elevation']} m ")
        print(f"INFO \t > Global POA irradiance: {(weather_data_poa['poa_global'] * 1).sum() / 1000 } kWh/m2/yr ")
        print(f"INFO \t > Diffuse POA irradiance: {(weather_data_poa['poa_diffuse'] * 1).sum() / 1000 } kWh/m2/yr ")
        print(f"INFO \t > Metadata: {meta} ")

        # Update the angles (usefull only for fixed mounting to calculate AOI losses)
        if self.installation['type'] == 'groundmounted_fixed':
            self._installation['tilt'] = meta['mounting_system']['fixed']['slope']['value']
            self._installation['azimuth'] = meta['mounting_system']['fixed']['azimuth']['value']
        else:
            self._installation['tilt'] = tilt
            self._installation['azimuth'] = azimuth

        return weather_data_poa

    def _create_pv_system(self):
        """Create a PV System with parameters compatible with the PVWatts model"""

        # Set moutning conditions for the thermal model
        mounting = 'freestanding'

        if self.installation['type'] == 'rooftop':
            mounting = 'insulated'

        system = pvsystem.PVSystem(
            module_parameters={
                'pdc0': self.pv_module['efficiency'] * 1000,  # Nominal DC power of 1 m2 of PV panel
                'gamma_pdc': self.pv_module['temperature_coefficient']  # Temperature coefficient (negative value)
            },
            inverter_parameters = {
                'pdc0': self.pv_module['efficiency'] * 1000,  # Nominal DC power
                'eta_inv_nom': 1.0,  # Inverter efficiency of 100% (system losses are computed ex-post)
                'ac_0': self.pv_module['efficiency'] * 1000  # AC power rating assumed equal to DC power rating
            },
            temperature_model_parameters = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['pvsyst'][mounting], # PVSyst temperature model    
            surface_tilt = self.installation['tilt'], # Used for AOI losses
            surface_azimuth = self.installation['azimuth'] # Used for AOI losses       
        )

        return system

    #######################################
    #### Location, Weather, PV System #####
    #######################################

    def compute_pv_production(self):
        """Compute the PV production and main KPIs using POA weather data"""

        print(f"INFO \t Computing the hourly PV production")

        # Include AOI losses if fixed tilt
        if self.installation['type'] == 'groundmounted_fixed':
            aoi_model='physical'
        else:
            aoi_model='no_loss'

        # Initialize the model chain and run from POA
        mc = modelchain.ModelChain(self.pv_system, self.location, aoi_model=aoi_model, spectral_model="no_loss")
        mc.run_model_from_poa(self.weather_data)

        # Correct the DC power for system losses to get AC production
        pv_production = mc.results.dc * (1 - self.installation['system_losses'])

        # Compute KPIs
        performance_ratio = pv_production / (self.pv_module['efficiency'] * self.weather_data['poa_global'])
        capacity_factor = pv_production / (self.pv_module['efficiency'] * 1000)
        operating_temperature = mc.results.cell_temperature

        # Create a DataFrame with the results
        results_df = pd.DataFrame({
            'PV Production (W/m2)': pv_production,
            'Performance Ratio': performance_ratio,
            'Capacity Factor': capacity_factor,
            'Temperature (C)': operating_temperature,
            'POA Irradiance (W/m2)': self.weather_data['poa_global']
        })

        print(f"INFO \t > Energy yield: {(pv_production * 1).sum() / 1000} kWh/m2/yr")
        print(f"INFO \t > Performance ratio: {(pv_production * 1).sum() / (self.pv_module['efficiency'] * self.weather_data['poa_global']).sum() }")
        print(f"INFO \t > Average capacity factor: {capacity_factor.mean()} ")

        return results_df

    #######################################
    ########### Helper methods ############
    #######################################

    def get_timezone(self, lat, lon):
        """Get timezone string based on latitude and longitude."""

        tf = TimezoneFinder() # Initialize TimezoneFinder

        if lat is not None and lon is not None:
            tz_string = tf.timezone_at(lat=lat, lng=lon)
            if tz_string:
                return tz_string
        else:
            return None       