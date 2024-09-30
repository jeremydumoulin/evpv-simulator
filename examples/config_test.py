# ------------------------------------------------------------------
# -- GENERAL -------------------------------------------------------
# ------------------------------------------------------------------

output_folder = "examples/output"
scenario_name = "AddisAbaba"

# ------------------------------------------------------------------
# -- EV DEMAND -----------------------------------------------------
# ------------------------------------------------------------------

# -- Required parameters

target_area_geojson = "examples/input/gadm41_ETH_1_AddisAbeba.json"
population_raster = "examples/input/GHS_POP_merged_4326_3ss_V1_0_R8andR9_C22_cropped.tif"
destinations_csv = "examples/input/workplaces.csv"
trips_per_inhabitant = 0.1
zone_width_km = 5
ORS_key = None

ev_fleet = [ ['car', 1.0], ['motorbike', 0.0] ]

charging_efficiency = 0.9

charging_home_share = 0.0
charging_home_arrivaltime_h = [18, 2]
charging_home_smart_share = 0.0

charging_destination_share = 1.0
charging_destination_arrivaltime_h = [9, 2]
charging_destination_smart_share = 0.0

# -- Optional parameters

# ------------------------------------------------------------------
# -- PV POTENTIAL --------------------------------------------------
# ------------------------------------------------------------------

# -- Required parameters

latitude = 9.005401
longitude = 38.763611
year = 2020

pv_module_efficiency = 0.22

installation = 'groundmounted_fixed'

# -- Optional parameters

# ------------------------------------------------------------------
# -- EV-PV SYNERGIES -----------------------------------------------
# ------------------------------------------------------------------

pv_capacity_MW = 280 