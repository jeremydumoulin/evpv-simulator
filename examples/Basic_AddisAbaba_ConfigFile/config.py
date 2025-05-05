# ------------------------------------------
# CONFIGURATION FILE: EV-PV MODEL
# ------------------------------------------
# Defines inputs for simulating EV charging demand,
# PV production, and EV-PV synergies.
# ------------------------------------------

# ========================================== #
# ============= BASIC PARAMETERS =========== #
# ========================================== #

# --- General ---
output_folder = "output"  # Output folder path (stores the output files)
scenario_name = "AddisAbaba"  # Scenario label (used in output filenames)

# --- EV Fleet Properties ---
fleet_config = {
    "total_vehicles": 1000,
    "vehicle_types": [
        {
            "name": "car",
            "battery_capacity_kWh": 50,
            "consumption_kWh_per_km": 0.18,
            "max_charging_power_kW": 100,
            "share": 0.9
        },
        {
            "name": "motorcycle",
            "battery_capacity_kWh": 10,
            "consumption_kWh_per_km": 0.05,
            "max_charging_power_kW": 7,
            "share": 0.1
        }
    ]
}

# --- Region of Interest ---
region_geojson = "input/gadm41_ETH_1_AddisAbeba.json"  # Boundary polygon (geojson)
population_raster = "input/GHS_POP_merged_4326_3ss_V1_0_R8andR9_C22_cropped.tif" # Population raster (.tif in WGS84 coordinate system)
workplaces_csv = "input/workplaces.csv"  # Workplace locations (CSV file)
pois_csv = "input/pois.csv"  # Points of interest (CSV file)
target_size_km = 5  # Target traffic zone size (in km)

# --- Mobility Demand Simulation ---
ors_key = None  # ORS API key (None if not using routing)
road_to_euclidian_ratio = 1.63  # Fallback ratio between distance by road and euclidian distance (if ORS not used)

# --- Charging Demand ---
# The charging scenario, with:
#   - "share": fraction of EVs charging at this location
#   - "power_options_kW": list of [charging power in kW, probability] pairs.
#   - "arrival_time_h": [mean, std] of arrival time in hours (optional for POIs, calculated automatically if not provided)
scenario = {
    "home": {
        "share": 0.5,
        "power_options_kW": [[3.7, 0.9], [7.4, 0.1]],
        "arrival_time_h": [18, 2]
    },
    "work": {
        "share": 0.3,
        "power_options_kW": [[7.4, 0.9], [11, 0.1]],
        "arrival_time_h": [9, 1]
    },
    "poi": {
        "share": 0.2,
        "power_options_kW": [[3.7, 0.1], [7.4, 0.3], [11, 0.6]]
        # Arrival time automatically calculated for POIs
    }
}

# --- PV Production ---
year = 2020 # Year to simulate
installation_type = "groundmounted_fixed"  
# Options: 'rooftop', 'groundmounted_fixed', 'groundmounted_singleaxis_horizontal', 
#          'groundmounted_singleaxis_vertical', 'groundmounted_dualaxis'

# --- EV-PV Complementarity ---
pv_capacity_MW = 10 # Installed nominal PV capacity
start_date = "01-01" # Start date of the simulation (Format: MM-DD)
end_date = "01-07" # End date of the simulation (Format: MM-DD)

# ========================================== #
# =========== ADVANCED PARAMETERS ========== #
# ========================================== #

# --- Traffic Zone Creation ---
crop_to_region = True  # Clip traffic zones to region of interest boundaries

# --- Mobility Model Settings ---
randomness = 0.0  # Add randomness to trip distribution results
model_type = "gravity_exp_scaled"  # Options: 'gravity_exp_scaled', 'radiation'
attraction_feature = "workplaces"  # Zone attractiveness feature. Options: 'population', 'workplaces'
cost_feature = "distance_road"  # Travel cost measure. Options: 'distance_road', 'distance_centroid'
distance_offset_km = 0.0  # Offset in the daily travel distance (added to the commuting distance)

# --- Charging Simulation ---
charging_efficiency = 0.9
time_step = 0.1  # Time resolution (in hours)

# --- PV Simulation Parameters ---
efficiency = 0.22  # PV module nominal efficiency
temperature_coefficient = -0.004  # Efficiency drop per °C
system_losses = 0.14  # Additional losse factor due to the PV system

# --- EV-PV complementarity ---
recompute_probability = 0.0  # Probability of recomputing daily charge needs from one day to another