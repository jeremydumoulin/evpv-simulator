# coding: utf-8

from evpv.vehicle import Vehicle
from evpv.vehiclefleet import VehicleFleet
from evpv.region import Region

car = Vehicle(name = "car", battery_capacity_kwh = 70, consumption_kwh_per_km = 0.2)
motorcycle = Vehicle(name = "motorcycle", battery_capacity_kwh = 10, consumption_kwh_per_km = 0.06)
fleet = VehicleFleet(total_vehicles = 100, vehicle_types = [[car, 0.5], [motorcycle, 0.5]])

print(fleet)

region = Region(
    region_geojson="examples/input/gadm41_ETH_1_AddisAbeba.json",    
    population_raster="examples/input/GHS_POP_merged_4326_3ss_V1_0_R8andR9_C22_cropped.tif",
    workplaces_csv="examples/input/workplaces.csv",
    pois_csv="examples/input/intermediate_stops.csv",
    traffic_zone_properties={
        "target_size_km": 3,
        "shape": "rectangle",
        "crop_to_region": True
    }
)

region.traffic_zone_properties={
        "target_size_km": 5,
        "shape": "rectangle",
        "crop_to_region": True
    }