# ------------------------------------------------------------------------------------------------------------------------
# CONFIGURATION FILE FOR THE EV-PV MODEL
# ------------------------------------------------------------------------------------------------------------------------
# This configuration file is used to set up the parameters required for computing the electric vehicle (EV) charging 
# demand, photovoltaic (PV) potential, and EV-PV synergies
# 
# The file is organized into two main sections:
#   1. REQUIRED PARAMETERS: These must be specified for the code to run.
#   2. OPTIONAL PARAMETERS: These can be customized or left as default.
# ------------------------------------------------------------


# ------------------------------------------------------------------------------------------------------------------------
# 1. REQUIRED PARAMETERS
# These parameters must be specified by the user for the code to run.
# ------------------------------------------------------------------------------------------------------------------------


# ------------------------------------------------------------
# 1.0 General Parameters
# General settings for the output location and scenario name.
# ------------------------------------------------------------
output_folder = "examples/output"     # Path to the folder where you want to store the output results (absolute or relative to the location of the evpv/ installation folder)
scenario_name = "AddisAbaba"          # Name of the scenario (e.g., city or region of your case study) - This is only used as a prefix for the simulation result files


# ------------------------------------------------------------
# 1.1 Electric Vehicle Charging Demand 
# Parameters related to the mobility demand, electric vehicle fleet, charging efficiency,
# and charging scenario to consider.
# ------------------------------------------------------------

# --- 1.1.1 Mobility Demand for Commuting ---
# Data used to compute the mobility demand for commuting purposes.

# Georeferenced data (paths to required input files)
target_area_geojson = "examples/input/gadm41_ETH_1_AddisAbeba.json"   # GeoJSON for the target area
population_raster = "examples/input/GHS_POP_merged_4326_3ss_V1_0_R8andR9_C22_cropped.tif"  # Raster file for population distribution
destinations_csv = "examples/input/workplaces.csv"  # CSV file with the location of all possible home to work destinations (e.g., workplaces from open street map)
intermediate_stops_csv = 'examples/input/intermediate_stops.csv'  # CSV file with the location of all possible stops between origin and destisations (e.g., POIs)

# Other mobility parameters
n_vehicles = 1000   # Number of vehicles
zone_width_km = 5            # Width of each traffic zone (in kilometers) for mobility demand
ORS_key = None               # OpenRouteService API key (set to None if not used)


# --- 1.1.2 Electric Vehicle Fleet ---
# Define the types of electric vehicles (EVs) and their associated share.

# EV fleet configuration: types of vehicles to consider and their respective share
# The types of vehicles have to be defined below
ev_fleet = lambda: [
    [globals()["vehicle_1"], 1.0],  # 100% of the fleet is vehicle_1
    [globals()["vehicle_2"], 0.0]   # 0% of the fleet is vehicle_2
]

# --- Definition of Electric Vehicle Properties ---
# Define the specifications of the electric vehicles.
# Feel free to add more vehicles by following the same structure.

# Vehicle 1: specifications (EV consumption, occupancy, charger power)
vehicle_1 = {
    'ev_consumption': 0.183,     # EV consumption in kWh/km
    'battery_capacity': 50,      # Nominal battery capacity in kWh
    'charger_power': {
        'Origin': [[7, 1.0]],     # Mix of charger power at origin (=home) in kW and probability to use that power (100%)
        'Destination': [[7, 1.0]], # Mix of charger power at destination in kW and probability (100%)
        'Intermediate': [[7, 1.0]] # Mix of charger power at intermediate stops in kW and probability (100%)        
    }
}

# Vehicle 2: specifications (EV consumption, occupancy, charger power)
vehicle_2 = {
    'ev_consumption': 0.13,      # EV consumption in kWh/km
    'battery_capacity': 40,      # Nominal battery capacity in kWh
    'charger_power': {
        'Origin': [[11, 0.5], [22, 0.5]],     # Mix of charger power at origin (=home) in kW and probability to use that power 
        'Destination': [[11, 0.5], [22, 0.5]], # Mix of charger power at charger power at destination in kW and probability
        'Intermediate': [[11, 0.5], [22, 0.5]] # Mix of charger power at intermediate stops in kW and probability         
    }
}


# --- 1.1.3 Charging Efficiency ---
charging_efficiency = 0.9        # Efficiency of charging (90%)


# --- 1.1.4 Charging Scenario ---
# Define the scenario for charging behavior (e.g., at home, at destination).

charging_scenario = {
    "Home": {
        "Share": 0.0,               # No charging at home (0%)
        "Arrival time": [18, 2],    # Mean and std deviation for arrival times at home
        "Smart charging": 0.0       # No smart charging at home
    },
    "Destination": {
        "Share": 1.0,              # 100% of charging at destination (e.g., workplaces)
        "Arrival time": [9, 2],    # Mean and std deviation for arrival times at destination
        "Smart charging": 0.0      # No smart charging at destination
    },
    "Intermediate": {
        "Share": 0.0,              # 0% of charging at intermediate stops (e.g., POIs)
        "Smart charging": 0.0      # No smart charging at intermediate stops
    }
}


# ------------------------------------------------------------
# 1.2 PV Potential
# Parameters to compute the photovoltaic potential for the target area.
# ------------------------------------------------------------

# Location of the PV installation (latitude and longitude)
latitude = 9.005401              # Latitude of the location
longitude = 38.763611            # Longitude of the location
year = 2020                      # Year for which the PV data is computed (between 2005 and 2020)

# PV system parameters
efficiency = 0.22                # Nominal efficiency of the PV system (22%)
installation = 'groundmounted_fixed'  # Installation type (options are: 'rooftop', 'groundmounted_fixed', 'groundmounted_dualaxis', 'groundmounted_singleaxis_horizontal', 'groundmounted_singleaxis_vertical')

# ------------------------------------------------------------
# 1.3 EV-PV Synergies
# Define the PV capacity and time period for the analysis.
# ------------------------------------------------------------
pv_capacity_MW = 280            # Total PV capacity in megawatts (MW)

# Time period for PV production analysis
start_date = '01-01'            # Start date of the analysis (MM-DD)
end_date = '01-30'              # End date of the analysis (MM-DD)


# ------------------------------------------------------------------------------------------------------------------------
# 2. OPTIONAL PARAMETERS
# These parameters are optional and can be adjusted for more advanced customization. If not specified, default values will
# be used.
# ------------------------------------------------------------------------------------------------------------------------
  
# ------------------------------------------------------------
# Electric Vehicle Charging Demand
# ------------------------------------------------------------ 

road_to_euclidian_ratio = 1.63 # Road to euclidian distance ratio (can be calculated for the target area using the script provided in the /scripts directory)
target_area_extension_km = 0.0 # Extension of the target area (km)
crop_zones_to_shapefile = True # Keep only Ttraffic zones inside the boundaries of the shapefile
spatial_interaction_model = 'gravity_exp_scaled' # Spatial interaction model to use (by default, self-calibrated gravity model is used)
attraction_feature = 'destinations' # Attraction feature for spatial interaction model ('destinations' or 'population')
cost_feature = 'distance_road' # Cost feature for spatial interaction model ('distance_road', 'distance_centroid', etc.)
vkt_offset =  0.0 # Additional daily distance travelled other than home to work commuting (km)
travel_time_origin_destination_h = 0.5 # Average travel time between origin and destination (h)
time_step_h = 0.1 # Time step for the charging curve (h)

# ------------------------------------------------------------
# PV Potential
# ------------------------------------------------------------

temperature_coefficient = -0.0035 # Relative efficiency loss per K
system_losses = 0.14 # System losses 