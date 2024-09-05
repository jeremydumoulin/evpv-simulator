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
        # Initialize the environment attributes
        self._environment = {
            'latitude': environment.get('latitude'),
            'longitude': environment.get('longitude'),
            'year': environment.get('year', '2020'),
            'timezone': environment.get('timezone', self.get_timezone(environment.get('latitude'), environment.get('longitude')))
        }

        # Initialize the installation attributes
        self._installation = {
            'tracking': installation.get('tracking', 'fixed'), # fixed, dualaxis, singleaxis_horizontal, singleaxis_vertical
            'tilt': installation.get('tilt', None),
            'azimuth': installation.get('azimuth', None),
            'mounting': installation.get('mounting', 'freestanding'), # freestanding or insulated
            'system_losses': installation.get('system_losses', 0.14)
        }

        # Initialize the PV device attributes
        self._pv_module = {
            'efficiency': pv_module.get('efficiency', 0.22),
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

        print(f"INFO \t Fetching hourly weather data with POA irrdiance from PV GIS for the year {self.environment['year']} - Tracking: {self.installation['tracking']}")

        # Initialize tilt and azimuth
        tilt = 0
        azimuth = 180
        optimize_tilt = optimize_azimuth = True

        # Set the tracking and tilt/azimuth options
        if self.installation['tracking'] == 'fixed':
            trackingtype = 0

            # Optimize or not tilt/azimuth angle depending on the user input
            if self.installation['tilt'] == None:            
                optimize_tilt = True
                print(f"INFO \t > No tilt angle provided. The optimum tilt will be estimated using PVGIS.")
            else:
                tilt = self.installation['tilt']
                optimize_tilt = False

            if self.installation['azimuth'] == None:                
                optimize_azimuth = True
                print(f"INFO \t > No azimuth angle provided. Both the optimum tilt and azimuth will be estimated using PVGIS.")
            else:
                azimuth = self.installation['azimuth']
                optimize_azimuth = False

        elif self.installation['tracking'] == 'singleaxis_horizontal':
            trackingtype = 1
        elif self.installation['tracking'] == 'singleaxis_vertical':
            trackingtype = 3
        elif self.installation['tracking'] == 'dualaxis':
            trackingtype = 2
        else:
            raise ValueError(f"Tracking type is unknown")

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
        weather_data_poa= weather_data_poa.tz_convert(self.environment['timezone'])

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

        # # Update the angles (usefull only for fixed mounting to calculate AOI losses)
        if self.installation['tracking'] == 'fixed' and (self.installation['tilt'] == None or self.installation['azimuth'] == None):
            self._installation['tilt'] = meta['mounting_system']['fixed']['slope']['value']
            self._installation['azimuth'] = meta['mounting_system']['fixed']['azimuth']['value']
        else:
            self._installation['tilt'] = tilt
            self._installation['azimuth'] = azimuth

        return weather_data_poa

    def _create_pv_system(self):
        """Create a PV System with parameters compatible with the PVWatts model"""

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
            temperature_model_parameters = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['pvsyst'][self.installation['mounting']], # PVSyst temperature model    
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
        if self.installation['tracking'] == 'fixed':
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