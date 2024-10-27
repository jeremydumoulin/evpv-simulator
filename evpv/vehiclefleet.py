# coding: utf-8

class VehicleFleet:
    """A class to represent a fleet of electric vehicles (EVs) with specified types and their shares."""

    def __init__(self, total_vehicles: int, vehicle_types: list):
        """
        Initializes the VehicleFleet class.

        Args:
            total_vehicles (int): The total number of vehicles in the fleet.
            vehicle_types (list): A list of pairs [Vehicle, share], where each share is a value between 0 and 1.            
        """
        self.vehicle_types = vehicle_types
        self.total_vehicles = total_vehicles

    # Properties and Setters
    @property
    def vehicle_types(self) -> list:
        """list: The list of vehicle types and their shares in the fleet."""
        return self._vehicle_types

    @vehicle_types.setter
    def vehicle_types(self, value: list):
        total_share = sum(share for _, share in value)
        if not all(0 <= share <= 1 for _, share in value):
            raise ValueError("Each vehicle share must be a positive value between 0 and 1.")
        if not total_share == 1:
            raise ValueError("The sum of vehicle shares must be equal to 1.")
        self._vehicle_types = value

    @property
    def total_vehicles(self) -> int:
        """int: The total number of vehicles in the fleet."""
        return self._total_vehicles

    @total_vehicles.setter
    def total_vehicles(self, value: int):
        if value <= 0:
            raise ValueError("Total number of vehicles must be a positive integer.")
        self._total_vehicles = value

    # Fleet metrics
    def average_battery_capacity(self) -> float:
        """Calculates the average battery capacity based on vehicle shares.

        Returns:
            float: The average battery capacity in kWh.
        """
        return sum(vehicle.battery_capacity_kwh * share for vehicle, share in self.vehicle_types)

    def average_consumption(self) -> float:
        """Calculates the average consumption based on vehicle shares.

        Returns:
            float: The average consumption in kWh/km.
        """
        return sum(vehicle.consumption_kwh_per_km * share for vehicle, share in self.vehicle_types)

    # Magic methods
    def __str__(self):
        """
        Returns a string representation of the VehicleFleet.

        Returns:
            str: A summary of the fleet composition and its class type.
        """
        vehicle_types_summary = ', '.join([f"{vehicle.name}: {share * 100:.1f}%" for vehicle, share in self.vehicle_types])
        return (f"VehicleFleet Object\n"
                f"  Total Vehicles: {self.total_vehicles}\n"
                f"  Vehicle Types: [{vehicle_types_summary}]")