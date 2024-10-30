# coding: utf-8

import json
import os
import rasterio
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import shape, LineString, Point, Polygon, box, MultiPoint
from shapely.ops import transform, nearest_points, snap
from geopy.distance import geodesic, distance
import openrouteservice
import time
import math
import csv
import pickle
import folium
import branca.colormap as cm

from evpv.vehicle import Vehicle
from evpv.vehiclefleet import VehicleFleet
from evpv.region import Region
from evpv.mobilitysimulator import MobilitySimulator
from evpv.pvsimulator import PVSimulator

class ChargingSimulator:
    """  
    A class to simulate the daily charging demand.
    """
    def __init__(self, region: Region, vehicle_fleet: VehicleFleet, mobility_demand: MobilitySimulator, scenario: dict):
        """
        Initializes the ChargingDemandSimulator class.
        """
        print("=========================================")
        print(f"INFO \t Creation of a MobilitySimulator object.")
        print("=========================================")

        self.vehicle_fleet = vehicle_fleet
        self.region = region
        self.mobility_demand = mobility_demand   
        self.scenario = scenario

        print(f"INFO \t Successful initialization of input parameters.")

        # Modeling results
        self._spatial_demand = pd.DataFrame()
        self._temporal_demand = pd.DataFrame() 

    # Results properties (read-only)

    @property
    def spatial_demand(self) -> pd.DataFrame:
        """pd.DataFrame: The spatial charging demand."""
        return self._spatial_demand

    @property
    def temporal_demand(self) -> pd.DataFrame:
        """pd.DataFrame: The temporal charging demand."""
        return self._temporal_demand

    # Properties and Setters 

    @property
    def region(self) -> Region:
        """Region: The region used in the simulation."""
        return self._region

    @region.setter
    def region(self, value: Region):
        if not isinstance(value, Region):
            raise ValueError("region must be an instance of the Region class.")
        self._region = value

    @property
    def vehicle_fleet(self) -> VehicleFleet:
        """VehicleFleet: The vehicle fleet used in the simulation."""
        return self._vehicle_fleet

    @vehicle_fleet.setter
    def vehicle_fleet(self, value: VehicleFleet):
        if not isinstance(value, VehicleFleet):
            raise ValueError("vehicle_fleet must be an instance of the VehicleFleet class.")
        self._vehicle_fleet = value

    @property
    def mobility_demand(self) -> MobilitySimulator:
        """MobilitySimulator: The mobility demand (i.e. flows between traffic zones) aggregated over all simulations provided."""
        return self._mobility_demand

    @mobility_demand.setter
    def mobility_demand(self, value: MobilitySimulator):
        if not isinstance(value, MobilitySimulator):
            raise ValueError(f"mobility_demand object must be an instance of the MobilitySimulator class.")
        if not 'n_outflows' in value.aggregated_zone_metrics.columns:
            raise ValueError("MobilitySimulator object - Vehicle allocation has not been performed.")
        # Check if trip distribution has been performed
        if not 'Origin' in value.flows.columns:
            raise ValueError("MobilitySimulator object - Trip distribution has not been performed.")

        self._mobility_demand = value

    @property
    def scenario(self) -> dict:
        """dict: The charging scenario."""
        return self._scenario

    @scenario.setter
    def scenario(self, value: dict):
        # Define required structure and expected types for each scenario
        required_keys = {
            'home': {
                'share': (float,),  # share should be a float
                'power_options': list,  # power_options should be a list
                'arrival_time': list,  # arrival_time should be a list
            },
            'work': {
                'share': (float,),
                'power_options': list,
                'arrival_time': list,
            },
            'pois': {
                'share': (float,),
                'power_options': list,
                'arrival_time': list,
            }
        }

        # Validate input value
        if not isinstance(value, dict):
            raise ValueError("scenario must be a dictionary.")

        total_share = 0  # To track the total share for validation

        for key, expected_structure in required_keys.items():
            if key not in value:
                raise ValueError(f"Missing required key: '{key}' in scenario.")

            # Check that each key's value is a dictionary and follows the expected structure
            scenario_part = value[key]
            if not isinstance(scenario_part, dict):
                raise ValueError(f"The value for '{key}' must be a dictionary.")

            # Validate each attribute in the scenario part
            for attr, expected_type in expected_structure.items():
                if attr not in scenario_part:
                    raise ValueError(f"Missing '{attr}' in '{key}' configuration.")
                if not isinstance(scenario_part[attr], expected_type):
                    raise TypeError(f"'{attr}' in '{key}' must be of type {expected_type}.")

            # Sum up the shares and validate they sum to 1
            total_share += scenario_part['share']
            
            # Validate power options format and shares
            power_options = scenario_part['power_options']
            if not all(isinstance(option, list) and len(option) == 2 and 
                       isinstance(option[0], (float, int)) and isinstance(option[1], (float, int)) 
                       for option in power_options):
                raise TypeError(f"Each 'power_options' entry in '{key}' must be a list of [power_level, share].")
            
            # Check that the sum of the shares in power_options is exactly 1
            if sum(option[1] for option in power_options) != 1:
                raise ValueError(f"The sum of shares in 'power_options' for '{key}' must be exactly 1.")
            
            # Validate arrival times are between 0 and 24
            arrival_time = scenario_part['arrival_time']
            if not all(isinstance(time, (float, int)) for time in arrival_time):
                raise TypeError(f"'arrival_time' in '{key}' must be a list of floats or integers.")
            if not all(0 <= time <= 24 for time in arrival_time):
                raise ValueError(f"All 'arrival_time' values in '{key}' must be between 0 and 24.")

        # Final check that total share is equal to 1
        if total_share != 1:
            raise ValueError("The sum of 'share' values across all keys must be exactly 1.")
        
        # If all checks pass, set the value
        self._scenario = value