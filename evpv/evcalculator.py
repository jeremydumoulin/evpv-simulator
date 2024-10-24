# coding: utf-8

from pathlib import Path
from evpv import helpers as hlp
from evpv.mobilitysim import MobilitySim
from evpv.chargingscenario import ChargingScenario

class EVCalculator:
    """
    A class to calculate electric vehicle (EV) mobility demand and charging demand based on provided inputs.
    
    Attributes:
        mobility_demand (dict): A dictionary containing parameters for mobility demand with the following keys:
            - 'target_area_geojson' (str): Path to the geojson containing the target area.
            - 'population_raster' (str): Path to population density raster (.tif).
            - 'destinations_csv' (str): Path to potential destinations CSV.
            - 'intermediate_stops_csv' (str): Path to potential intermediate stops CSV.
            - 'trips_per_inhabitant' (float): Average number of trips per inhabitant.
            - 'zone_width_km' (float): Spatial resolution of the zones (in km).
            - 'ORS_key' (str): Open Route Service (ORS) API key if needed. Defaults to None.
            - 'road_to_euclidian_ratio' (float, optional): Road to euclidian distance ratio. Defaults to 1.63.
            - 'target_area_extension_km' (float, optional): Extension of the target area (km). Defaults to 0.0.
            - 'crop_zones_to_shapefile' (bool, optional): If True, delete traffic zones outside the boundaries of the target area. Defaults to True.
            - 'spatial_interaction_model' (str, optional): Spatial interaction model. Defaults to 'gravity_exp_scaled'.
            - 'attraction_feature' (str, optional): Attraction feature ('destinations' or 'population'). Defaults to 'destinations'.
            - 'cost_feature' (str, optional): Cost feature ('distance_road', 'distance_centroid', etc.). Defaults to 'distance_road'.
            - 'km_per_capita_offset' (float, optional): Additional daily distance travelled (km). Defaults to 0.0.

        ev_fleet (list): A list defining the electric vehicle fleet and its share in the form [[vehicle1, share1], [vehicle2, share2], ...].

        charging_efficiency (float): Efficiency of the charging process (between 0 and 1).

        charging_scenario (dict): A dictionary containing parameters for charging scenarios with the following keys:
            - 'Home' (dict): Dictionary for home charging settings.
            - 'Destination' (dict): Dictionary for destination charging settings.
            - 'travel_time_origin_destination_h' (float, optional): Travel time between origin and destination in hours. Defaults to 0.5.
            - 'time_step_h' (float, optional): Time step for charging curve in hours. Defaults to 0.1.
    """
    
    #######################################
    ############# Constructor #############
    #######################################

    def __init__(self, mobility_demand: dict, ev_fleet: list, charging_efficiency: float, charging_scenario: dict):
        """
        Initializes EVCalculator with mobility demand, EV fleet, charging efficiency, and charging scenario.
        
        Args:
        mobility_demand (dict): A dictionary containing parameters for mobility demand with the following keys:
            - 'target_area_geojson' (str): Path to the geojson containing the target area.
            - 'population_raster' (str): Path to population density raster (.tif).
            - 'destinations_csv' (str): Path to potential destinations CSV.
            - 'intermediate_stops_csv' (str): Path to potential intermediate stops CSV.
            - 'trips_per_inhabitant' (float): Average number of trips per inhabitant.
            - 'zone_width_km' (float): Spatial resolution of the zones (in km).
            - 'ORS_key' (str, optional): Open Route Service (ORS) API key. Defaults to None.
            - 'road_to_euclidian_ratio' (float, optional): Road to euclidian distance ratio. Defaults to 1.63.
            - 'target_area_extension_km' (float, optional): Extension of the target area (km). Defaults to 0.0.
            - 'crop_zones_to_shapefile' (bool, optional): If True, delete traffic zones outside the boundaries of the target area. Defaults to True.
            - 'spatial_interaction_model' (str, optional): Spatial interaction model. Defaults to 'gravity_exp_scaled'.
            - 'attraction_feature' (str, optional): Attraction feature ('destinations' or 'population'). Defaults to 'destinations'.
            - 'cost_feature' (str, optional): Cost feature ('distance_road', 'distance_centroid', etc.). Defaults to 'distance_road'.
            - 'km_per_capita_offset' (float, optional): Additional daily distance travelled (km). Defaults to 0.0.

        ev_fleet (list): A list defining the electric vehicle fleet and its share in the form [[vehicle1, share1], [vehicle2, share2], ...].

        charging_efficiency (float): Efficiency of the charging process (between 0 and 1).

        charging_scenario (dict): A dictionary containing parameters for charging scenarios with the following keys:
            - 'Home' (dict): Dictionary for home charging settings.
            - 'Destination' (dict): Dictionary for destination charging settings.
            - 'travel_time_origin_destination_h' (float, optional): Travel time between origin and destination in hours. Defaults to 0.5.
            - 'time_step_h' (float, optional): Time step for charging curve in hours. Defaults to 0.1.
        """

        # Initialize the mobility demand attributes
        self._mobility_demand = {
            'target_area_geojson': mobility_demand.get('target_area_geojson'), # Path to the geojson containing the target area
            'population_raster': mobility_demand.get('population_raster'), # Path to population density raster (.tif)
            'destinations_csv': mobility_demand.get('destinations_csv'), # Path to potential destinations csv
            'intermediate_stops_csv': mobility_demand.get('intermediate_stops_csv'), # Path to potential intermediate_stops csv            
            'trips_per_inhabitant': mobility_demand.get('trips_per_inhabitant'), # Average number of trips per inhabitant
            'zone_width_km': mobility_demand.get('zone_width_km'), # Spatial resolution of the zones (in km)
            'ORS_key': mobility_demand.get('ORS_key'), # Open Route Service (ORS) API key. Set to None if you do not want to use ORS.

            # Optional parameters            
            'road_to_euclidian_ratio': mobility_demand.get('road_to_euclidian_ratio', 1.63), # Road to euclidian distance ratio
            'target_area_extension_km': mobility_demand.get('target_area_extension_km', 0.0), # Extension of the target area (km)
            'crop_zones_to_shapefile': mobility_demand.get('crop_zones_to_shapefile', True), # Crop zones to zones inside the target area
            'spatial_interaction_model': mobility_demand.get('spatial_interaction_model', 'gravity_exp_scaled'), # Spatial interaction model
            'attraction_feature': mobility_demand.get('attraction_feature', 'destinations'), # Attraction feature ('destinations' or 'population')
            'cost_feature': mobility_demand.get('cost_feature', 'distance_centroid'), # Cost feature ('distance_road', 'distance_centroid', etc.)
            'km_per_capita_offset': mobility_demand.get('km_per_capita_offset', 0.0), # Additional daily distance travelled (km)
        }

        # Initialize the EV fleet attributes
        self._ev_fleet = ev_fleet # EV Fleet in the form [[vehicle1, share1], [vehicle2, share2], ...]

        # Initialize the charging efficiency attribute
        self._charging_efficiency = charging_efficiency # Charging efficiency between 0 and 1

        # Initialize the charging curve attributes
        self._charging_scenario = {
            'Home': charging_scenario.get('Home'), # Dictionary for home charging
            'Destination': charging_scenario.get('Destination'), # Dictionary for destination charging
            "Intermediate": charging_scenario.get('Intermediate'),

            # Optional parameters
            'travel_time_origin_destination_h': charging_scenario.get('travel_time_origin_destination_h', 0.5), # Travel time between origin and destination
            'time_step_h': charging_scenario.get('time_step_h', 0.1), # Time step for charging curve
        }
        

    #######################################
    ######## Run the simulation ###########
    #######################################

    def compute_ev_demand(self):
        """
        Computes the mobility demand simulation and the EV demand according to the provided inputs
        """
        # Create MobilitySim object and run the mobility demand modeling
        self.mobsim = self._create_mobsim()
        self._compute_mobility_demand()

        # Create ChargingScenario object and run the scenario
        self.charging_demand = self._create_charging_scenario()
        self.charging_demand.spatial_charging_demand()
        self.charging_demand.temporal_charging_demand()

    #######################################
    ########### EV presets ################
    #######################################

    preset = {
        'car': {
            'ev_consumption': 0.183,
            'battery_capacity': 50,
            'vehicle_occupancy': 1.4,  
            'charger_power': {
                'Origin': [[3.3, 1.0]],
                'Destination': [[6.6, 1.0]],
                'Intermediate': [[6.6, 0.5], [44, 0.5]]                
            }
        },
        'motorbike': {
            'ev_consumption': 0.058,
            'battery_capacity': 10,
            'vehicle_occupancy': 1.0,  
            'charger_power': {
                'Origin': [[3, 1.0]],
                'Destination': [[3, 1.0]],
                'Intermediate': [[3, 1.0]]                
            }
        }  
    }  
    """Presets for electric vehicles (EVs).

    Class-level attributes containing default values (electric energy consumption, vehicle occupancy, possible charging powers) for different types of EVs. 

    Attributes:
        preset (dict): A dictionary containing presets for various types of EVs:
            - 'car': A dictionary with the following keys:
                - 'ev_consumption' (float): Average energy consumption (https://doi.org/10.1016/j.trd.2017.04.013) of the vehicle (kWh/km), value: 0.183.
                - 'vehicle_occupancy' (float): Average number of occupants per vehicle (ref. https://www.energy.gov/eere/vehicles/articles/fotw-1333-march-11-2024-2022-average-number-occupants-trip-household), value: 1.4.
                - 'charger_power' (dict): Charging power characteristics (ref.: https://doi.org/10.1016/j.rser.2023.114214):
                    - 'Origin': A list of lists containing charging power available at origin: [[7, 0.68], [11, 0.3], [22, 0.02]].
                    - 'Destination': A list of lists containing charging power available at destination: [[7, 0.68], [11, 0.3], [22, 0.02]].
            - 'motorbike': A dictionary with the following keys:
                - 'ev_consumption' (float): Average energy consumption (https://doi.org/10.3390/en16176369) of the motorcycle (kWh/km), value: 0.058.
                - 'vehicle_occupancy' (float): Average number of occupants per motorcycle, value: 1.0.
                - 'charger_power' (dict): Charging power characteristics:
                    - 'Origin': A list of lists containing charging power available at origin: [[7, 1.0]].
                    - 'Destination': A list of lists containing charging power available at destination: [[7, 1.0]].
                    - 'Intermediate': A list of lists containing charging power available at intermediate stops: [[7, 1.0]].
    """


    #######################################
    ### Parameters Setters and Getters ####
    #######################################

    @property
    def mobility_demand(self) -> dict:
        """
        Returns the mobility demand dictionary.
        
        Returns:
            dict: Mobility demand dictionary.
        """
        return self._mobility_demand

    @mobility_demand.setter
    def mobility_demand(self, value: dict):
        """
        Sets the mobility demand dictionary.

        Args:
            value (dict): Dictionary containing mobility demand values.
        """
        if isinstance(value, dict):
            self._mobility_demand.update(value)
        else:
            raise ValueError("mobility_demand must be a dictionary")

    @property
    def ev_fleet(self) -> list:
        """
        Returns the EV fleet list.
        
        Returns:
            list: List defining the EV fleet.
        """
        return self._ev_fleet

    @ev_fleet.setter
    def ev_fleet(self, value: list):
        """
        Sets the EV fleet list.

        Args:
            value (list): List of EV fleet data.
        """
        if isinstance(value, list):
            self._ev_fleet.update(value)
        else:
            raise ValueError("ev_fleet must be a list")

    @property
    def charging_efficiency(self) -> float:
        """
        Returns the charging efficiency value.
        
        Returns:
            float: Charging efficiency.
        """
        return self._charging_efficiency

    @charging_efficiency.setter
    def charging_efficiency(self, value: float):
        """
        Sets the charging efficiency value.

        Args:
            value (float): Charging efficiency value.
        """
        self._charging_efficiency = value

    @property
    def charging_scenario(self) -> dict:
        """
        Returns the charging scenario dictionary.
        
        Returns:
            dict: Charging scenario dictionary.
        """
        return self._charging_scenario

    @charging_scenario.setter
    def charging_scenario(self, value: dict):
        """
        Sets the charging scenario dictionary.

        Args:
            value (dict): Charging scenario dictionary.
        """
        if isinstance(value, dict):
            self._charging_scenario.update(value)
        else:
            raise ValueError("charging_scenario must be a dictionary")

    #######################################
    ########### Mobility Demand ###########
    #######################################

    def _create_mobsim(self) -> MobilitySim:
        """
        Creates a MobilitySim object using the mobility demand data.

        Returns:
            MobilitySim: Initialized MobilitySim object.
        """
        return MobilitySim(
            target_area=self.mobility_demand['target_area_geojson'],
            population_density=self.mobility_demand['population_raster'], 
            destinations=self.mobility_demand['destinations_csv'],
            intermediate_stops=self.mobility_demand['intermediate_stops_csv']
        )

    def _compute_mobility_demand(self):
        """
        Sets up and computes the mobility demand using the MobilitySim object.
        """
        self.mobsim.setup_simulation(
            taz_target_width_km=self.mobility_demand['zone_width_km'], 
            simulation_area_extension_km=self.mobility_demand['target_area_extension_km'], 
            crop_zones_to_shapefile=self.mobility_demand['crop_zones_to_shapefile']
        )

        self.mobsim.trip_generation(n_trips_per_inhabitant=self.mobility_demand['trips_per_inhabitant'])

        self.mobsim.trip_distribution(
            model=self.mobility_demand['spatial_interaction_model'], 
            ors_key=self.mobility_demand['ORS_key'], 
            attraction_feature=self.mobility_demand['attraction_feature'], 
            cost_feature=self.mobility_demand['cost_feature'], 
            km_per_capita_offset=self.mobility_demand['km_per_capita_offset']
        )

    #######################################
    ########## Charging Scenario ##########
    #######################################

    def _create_charging_scenario(self) -> ChargingScenario:
        """
        Creates a ChargingScenario object using mobility and charging data.

        Returns:
            ChargingScenario: Initialized ChargingScenario object.
        """
        return ChargingScenario(
            mobsim=[self.mobsim],
            ev_fleet=self.ev_fleet,
            charging_efficiency=self.charging_efficiency, 
            time_step=self.charging_scenario['time_step_h'], 
            scenario_definition={
                "Travel time origin-destination": self.charging_scenario['travel_time_origin_destination_h'],
                "Origin": self.charging_scenario['Home'],
                "Destination": self.charging_scenario['Destination'] ,
                "Intermediate": self.charging_scenario['Intermediate']               
            }
        )


    #######################################
    ########### Storing results ###########
    #######################################

    def save_results(self, output_folder: str, prefix: str, maps: bool = True):
        """
        Saves the results of the mobility and charging demand simulations to the specified folder.

        Args:
            output_folder (str): The path to the folder where results will be saved.
            prefix (str): A prefix for the file names to distinguish between different simulations.
            maps (bool, optional): Whether to generate and save maps of the results. Defaults to True.

        Returns:
            None
        """
        print(f"INFO \t SAVING RESULTS")

        output_folder = Path(output_folder)

        # Mobility demand
        self.mobsim.flows.to_csv(output_folder / f"{prefix}_MobilityDemand_OriginDestinationFlows.csv", index=False)
        self.mobsim.traffic_zones.to_csv(output_folder / f"{prefix}_MobilityDemand_TrafficAnalysisZones.csv", index=False)

        # Store vehicle-kilometers traveled (VKT) distribution
        vkt_distribution = self.mobsim.km_per_capita_histogram(bin_width_km=1)
        vkt_distribution.to_csv(output_folder / f"{prefix}_MobilityDemand_VKTdistribution.csv", index=False)

        if maps:
            # Save the setup and trip generation maps as HTML files
            self.mobsim.setup_to_map().save(output_folder / f"{prefix}_MobilityDemand_SimulationSetup.html")
            self.mobsim.trip_generation_to_map().save(output_folder / f"{prefix}_MobilityDemand_TripGeneration.html")

        # Charging demand
        self.charging_demand.charging_demand.to_csv(output_folder / f"{prefix}_ChargingDemand_Spatial.csv", index=False)
        self.charging_demand.charging_profile.to_csv(output_folder / f"{prefix}_ChargingDemand_ChargingCurve.csv", index=False)

        if maps:
            # Save charging demand maps as HTML files
            self.charging_demand.chargingdemand_total_to_map().save(output_folder / f"{prefix}_ChargingDemand_Total.html")
            self.charging_demand.chargingdemand_pervehicle_to_map().save(output_folder / f"{prefix}_ChargingDemand_PerCar.html")
            self.charging_demand.chargingdemand_nvehicles_to_map().save(output_folder / f"{prefix}_ChargingDemand_VehiclesToCharge.html")
