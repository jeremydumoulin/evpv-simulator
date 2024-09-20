# coding: utf-8

from pathlib import Path

from evpv import helpers as hlp
from evpv.mobilitysim import MobilitySim
from evpv.chargingscenario import ChargingScenario

class EVCalculator:
    #######################################
    ############# Constructor #############
    #######################################

    def __init__(self, mobility_demand: dict, ev_fleet: list, charging_efficiency: float, charging_scenario: dict):

        # Initialize the mobility demand attributes
        self._mobility_demand = {
            # REQUIRED
            'target_area_geojson': mobility_demand.get('target_area_geojson'), # Path to the geojson containing the path to the target area (or region of interest)
            'population_raster': mobility_demand.get('population_raster'), # Path to raster file (.tif) with the population density
            'destinations_csv': mobility_demand.get('destinations_csv'), # Path to the csv file with the list of potential destinations
            'trips_per_inhabitant': mobility_demand.get('trips_per_inhabitant'), # Average number of trips per inhabitant (from origin to destination, e.g., home to Destination)
            'zone_width_km': mobility_demand.get('zone_width_km'), # Target width (in km) of the zones that mesh the simulation zone (i.e., spatial resolution) - will be slighlty adapted by the algorithm

            # OPTIONAL
            'ORS_key': mobility_demand.get('ORS_key', None), # Open Route Service (ORS) API key. If no key is provided, the distance by road is estimated using an empirical ratio set by the user
            'road_to_euclidian_ratio': mobility_demand.get('road_to_euclidian_ratio', 1.63), # Empirical ratio between the distance by road and the euclidian distance (distance as the crow flies) 
            'target_area_extension_km': mobility_demand.get('target_area_extension_km', 0.0), # Extension of the the target area to include also in- and out- flows from outside
            'population_to_ignore_share': mobility_demand.get('population_to_ignore_share', 0.0), # Share of the population to ignore (will speed up calculation by ignoring sparsely populated zones)
            'spatial_interaction_model': mobility_demand.get('spatial_interaction_model', 'gravity_exp_scaled'), # Type of spatial interaction model to use ('gravity_exp_scaled' = autocalibrated gravity model)
            'attraction_feature': mobility_demand.get('attraction_feature', 'destinations'), # Attraction feature used in the spatial interaction model ('destinations', 'population')
            'cost_feature': mobility_demand.get('cost_feature', 'distance_road'), # Cost feature used in the spatial interaction model ('distance_road', 'distance_centroid', 'time_road')
            'km_per_capita_offset': mobility_demand.get('km_per_capita_offset', 0.0), # Additionnal daily distance travelled (in km) from the origin to destination (one way)      
        }

        # Initialize the EV fleet attributes
        self._ev_fleet = ev_fleet # EV Fleet in the form [[vehicle1, share1], [vehicle2, share2], ...]

        # Initialize the charging efficiency attribute
        self._charging_efficiency = charging_efficiency # Charging efficiency between 0 and 1

        # Initialize the charging curve attributes
        self._charging_scenario = {
            # REQUIRED
            'Home': charging_scenario.get('Home'), # Dictionnary containing the main parameters for charging at origin (home)
            'Destination': charging_scenario.get('Destination'), # Dictionnary containing the main parameters for charging at destination (eg work)

            # OPTIONAL
            'travel_time_origin_destination_h': charging_scenario.get('travel_time_origin_destination_h', 0.5), # Average travel time (in hours) form origin to/from destination (used for smart charging)
            'time_step_h': charging_scenario.get('time_step_h', 0.1), # Time step (in hours) for the charging curve
        }

        # Create MobilitySim object and run the mobility demand modelling
        self.mobsim = self._create_mobsim()
        self.compute_mobility_demand()
        print("")

        # Create ChargingScenario object and run the scenario
        self.charging_demand = self._create_charging_scenario()
        self.charging_demand.spatial_charging_demand()
        self.charging_demand.temporal_charging_demand()
        print("")

    #######################################
    ########### EV presets ################
    #######################################

    preset = {
        'car': {
        # Charging mix: https://doi.org/10.1016/j.rser.2023.114214
        # Car occupancy: https://www.energy.gov/eere/vehicles/articles/fotw-1333-march-11-2024-2022-average-number-occupants-trip-household
        # Electric consumption: https://doi.org/10.1016/j.trd.2017.04.013
            'ev_consumption': 0.183,
            'vehicle_occupancy': 1.4,  
            'charger_power': {
                'Origin': [[7, 0.68], [11, 0.3], [22, 0.02]],
                'Destination': [[7, 0.68], [11, 0.3], [22, 0.02]]
            }
        },
        'motorbike': {
        # Electric consumption: https://doi.org/10.3390/en16176369
            'ev_consumption': 0.058,
            'vehicle_occupancy': 1.0,  
            'charger_power': {
                'Origin': [[7, 1.0]],
                'Destination': [[7, 1.0]]
            }
        }  
    }  

    #######################################
    ### Parameters Setters and Getters ####
    #######################################

    @property
    def mobility_demand(self):
        return self._mobility_demand

    @mobility_demand.setter
    def mobility_demand(self, value):
        if isinstance(value, dict):
            self._mobility_demand.update(value)
        else:
            raise ValueError("mobility_demand must be a dictionary")

    @property
    def ev_fleet(self):
        return self._ev_fleet

    @ev_fleet.setter
    def ev_fleet(self, value):
        if isinstance(value, list):
            self._ev_fleet.update(value)
        else:
            raise ValueError("ev_fleet must be a list")

    @property
    def charging_efficiency(self):
        return self._charging_efficiency

    @charging_efficiency.setter
    def charging_efficiency(self, value):        
        self._charging_efficiency.update(value)

    @property
    def charging_scenario(self):
        return self._charging_scenario

    @charging_scenario.setter
    def charging_scenario(self, value):
        if isinstance(value, dict):
            self._charging_scenario.update(value)
        else:
            raise ValueError("charging_scenario must be a dictionary")

    #######################################
    ########### Mobility Demand ###########
    #######################################

    def _create_mobsim(self):
        return MobilitySim(
            target_area = self.mobility_demand['target_area_geojson'],
            population_density = self.mobility_demand['population_raster'], 
            destinations = self.mobility_demand['destinations_csv']
        )

    def compute_mobility_demand(self):
        self.mobsim.setup_simulation(
            taz_target_width_km = self.mobility_demand['zone_width_km'], 
            simulation_area_extension_km = self.mobility_demand['target_area_extension_km'], 
            population_to_ignore_share = self.mobility_demand['population_to_ignore_share'])

        self.mobsim.trip_generation(n_trips_per_inhabitant = self.mobility_demand['trips_per_inhabitant']) 

        self.mobsim.trip_distribution(
            model = self.mobility_demand['spatial_interaction_model'], 
            ors_key = self.mobility_demand['ORS_key'], 
            attraction_feature = self.mobility_demand['attraction_feature'], 
            cost_feature = self.mobility_demand['cost_feature'], 
            km_per_capita_offset = self.mobility_demand['km_per_capita_offset'])

    #######################################
    ########## Charging Scenario ##########
    #######################################

    def _create_charging_scenario(self):
        return ChargingScenario(
            mobsim = [self.mobsim],
            ev_fleet = self.ev_fleet,
            charging_efficiency = self.charging_efficiency, 
            time_step = self.charging_scenario['time_step_h'], 
            scenario_definition = {
                "Travel time origin-destination": self.charging_scenario['travel_time_origin_destination_h'], 
                "Origin": self.charging_scenario['Home'],
                "Destination": self.charging_scenario['Destination']
            }
        )

    #######################################
    ########### Storing results ###########
    #######################################

    def save_results(self, output_folder, prefix, maps = True):
        print(f"INFO \t SAVING RESULTS")

        output_folder = Path(output_folder)

        # Mobility demand
        self.mobsim.flows.to_csv(output_folder / f"{prefix}_MobilityDemand_OriginDestinationFlows.csv", index=False) # Store aggregated TAZ features as csv
        self.mobsim.traffic_zones.to_csv(output_folder / f"{prefix}_MobilityDemand_TrafficAnalysisZones.csv", index=False) # Store aggregated TAZ features as csv

        vkt_distribution = self.mobsim.km_per_capita_histogram(bin_width_km = 1)
        vkt_distribution.to_csv(output_folder / f"{prefix}_MobilityDemand_VKTdistribution.csv", index=False)

        if maps:
            self.mobsim.setup_to_map().save(output_folder  / f"{prefix}_MobilityDemand_SimulationSetup.html")
            self.mobsim.trip_generation_to_map().save(output_folder / f"{prefix}_MobilityDemand_TripGeneration.html")

        # Charging demand

        self.charging_demand.charging_demand.to_csv(output_folder  / f"{prefix}_ChargingDemand_Spatial.csv", index=False) 
        self.charging_demand.charging_profile.to_csv(output_folder  / f"{prefix}_ChargingDemand_ChargingCurve.csv", index=False)

        if maps:
            self.charging_demand.chargingdemand_total_to_map().save(output_folder  / f"{prefix}_ChargingDemand_Total.html")
            self.charging_demand.chargingdemand_pervehicle_to_map().save(output_folder  / f"{prefix}_ChargingDemand_PerCar.html")
            self.charging_demand.chargingdemand_nvehicles_to_map().save(output_folder  / f"{prefix}_ChargingDemand_VehiclesToCharge.html")  