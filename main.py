# coding: utf-8

from evpv.vehicle import Vehicle
from evpv.vehiclefleet import VehicleFleet

car = Vehicle(name = "car", battery_capacity_kwh = 70, consumption_kwh_per_km = 0.2)
motorcycle = Vehicle(name = "motorcycle", battery_capacity_kwh = 10, consumption_kwh_per_km = 0.06)

fleet = VehicleFleet(total_vehicles = 100, vehicle_types = [[car, 0.5], [motorcycle, 0.5]])

print(fleet.average_battery_capacity())