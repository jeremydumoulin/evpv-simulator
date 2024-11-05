# ------------------------------------------------------------------------------------------------------------------------
# CONFIGURATION FILE FOR THE EV-PV MODEL
# ------------------------------------------------------------------------------------------------------------------------
# This configuration file is used to set up the parameters required for computing the electric vehicle (EV) charging 
# demand, photovoltaic (PV) potential, and EV-PV synergies
# 
# The file is organized into two main sections:
#   1. BASIC PARAMETERS: These must be specified for the code to run.
#   2. ADVANCED PARAMETERS: These can be customized or left as default.
# ------------------------------------------------------------


# ------------------------------------------------------------------------------------------------------------------------
# 1. BASIC PARAMETERS
# These parameters must be specified by the user for the code to run.
# ------------------------------------------------------------------------------------------------------------------------


# ------------------------------------------------------------
# 1.0 General Parameters
# ------------------------------------------------------------
output_folder = "examples/output"     # Path to the folder where you want to store the output results (absolute or relative to the location of the evpv/ installation folder)
scenario_name = "AddisAbaba"          # Name of the scenario (e.g., city or region of your case study) - This is only used as a prefix for the simulation result files

# ------------------------------------------------------------
# 1.1 Electric Vehicle Fleet
# ------------------------------------------------------------

# Individual electric vehicle properties and shares in the total fleet
ev_1 = {
    "name": "car",
    "battery_capacity_kWh": 50,
    "consumption_kWh_per_km": 0.18,
    "max_charging_power_kW": 22,
    "share": 0.9
}

ev_2 = {
    "name": "motorcycle",
    "battery_capacity_kWh": 10,
    "consumption_kWh_per_km": 0.05,
    "max_charging_power_kW": 10,
    "share": 0.1
}

# Number of electric vehicles to simulate
fleet_config = {
    "total_vehicles": 1000,
    "vehicle_types": [ev_1, ev_2]
}

# ------------------------------------------------------------
# 1.2 Region of Interest
# ------------------------------------------------------------

# Georeferenced data (paths to required input files)
region_geojson = "examples/input/gadm41_ETH_1_AddisAbeba.json"   # GeoJSON for the target area
population_raster = "examples/input/GHS_POP_merged_4326_3ss_V1_0_R8andR9_C22_cropped.tif"  # Raster file for population distribution
workplaces_csv = "examples/input/workplaces.csv"  # CSV file with the location of all possible home to work destinations (e.g., workplaces from open street map)
pois_csv = 'examples/input/pois.csv'  # CSV file with the location of all possible POIs 

# Traffic zone properties
target_size_km = 5 # Target width of each traffic zone (in kilometers) 

# ------------------------------------------------------------
# 1.3 Mobility Demand Simulation
# ------------------------------------------------------------

ors_key = None # None or ORS Key to perform routing between traffic zones using open route service
road_to_euclidian_ratio = 1.63 # Average ratio between the distance by road and the distance as the crows flies (typically around 1.5)

# ------------------------------------------------------------
# 1.4 Charging Demand
# ------------------------------------------------------------

# Charging scenario (location, available power options for charging, arrival time at charging location)
scenario = {
    'home': {
        'share': 0.5,  # 50% of EVs charge at home
        'power_options_kW': [[3.7, 0.9], [7.4, 0.1]],    
        'arrival_time_h': [18, 2]  # Arrival time with mean and std deviation
    },
    'work': {
        'share': 0.3,  # 30% of EVs charge at work
        'power_options_kW': [[7.4, 0.9], [11, 0.1]],    
        'arrival_time_h': [9, 1]
    },
    'poi': {
        'share': 0.2,  # 20% of EVs charge at points of interest
        'power_options_kW': [[3.7, 0.1], [7.4, 0.3], [11, 0.6]]    
    }
}

# ------------------------------------------------------------
# 1.5 PV Production Potential
# ------------------------------------------------------------

year = 2020 # Reference year                  
installation_type = 'groundmounted_fixed'  # Installation type (options are: 'rooftop', 'groundmounted_fixed', 'groundmounted_dualaxis', 'groundmounted_singleaxis_horizontal', 'groundmounted_singleaxis_vertical')

# ------------------------------------------------------------
# 1.6 EV-PV Synergies
# ------------------------------------------------------------

pv_capacity_MW = 10

start_date = '01-01'            # Start date of the analysis (MM-DD)
end_date = '01-07'              # End date of the analysis (MM-DD)


# ------------------------------------------------------------------------------------------------------------------------
# 2. ADVANCED PARAMETERS
# These parameters must be specified by the user for the code to run.
# ------------------------------------------------------------------------------------------------------------------------


zone_shape = "rectangle"
crop_to_region = True

allocation_method = "population"
randomness = 0.0
model_type = "gravity_exp_scaled"               
attraction_feature = "workplaces"    
cost_feature = "distance_road"       
distance_offset_km = 0.0    

charging_efficiency = 0.9
time_step = 0.1

efficiency = 0.22 # Nominal efficiency  
temperature_coefficient = -0.004
system_losses = 0.14

recompute_probability = 0.0 