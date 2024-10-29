# coding: utf-8

import json
import os
import rasterio
from rasterio.mask import mask
import csv
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import shape, LineString, Point, Polygon, box, MultiPoint
from shapely.ops import transform, nearest_points, snap
from geopy.distance import geodesic, distance
from pyproj import Geod
import folium
import branca.colormap as cm

class Region:
    """
    A class to represent the region of interest, subdivided into traffic zones populated with aggregated geospatial data.

    This class enables geospatial data input validation, subdivision into traffic zones, and the aggregation of data 
    across these zones. Specifically, it facilitates the processing of population data from raster files, workplace and POI data 
    from CSVs, and the generation of traffic zones based on configurable properties. The class also supports the export of traffic 
    zone data for further analysis and visualization through CSV files and Folium maps.

    Key Features:
    - Input Data Validation: Validates input data and recalculates/repopulates traffic zones on-the-fly when key properties are modified.
    - Traffic Zone Creation: Defines the traffic zones based of zone target size and shape, and provides the option to 
      crop zones to the boundaries of the defined region.
    - Data Aggregation: Aggregates population, workplace, and POI data within each traffic zone.
    - Export and Visualizatio: Offers traffic zone export to CSV and visualization via Folium maps

    Note: The region of interest is converted to a bounding box for zoning purposes. Set the `crop_to_region` parameter to `True` to retain only 
    zones within the region's boundaries.
    """

    def __init__(self, region_geojson: str, population_raster: str, workplaces_csv: str, pois_csv: str, traffic_zone_properties: dict):
        """
        Initializes the Region class with region data and properties for traffic zoning.

        Args:
            region_geojson (str): Path to the region geojson file.
            population_raster (str): Path to the population density raster file.
            workplaces_csv (str): Path to the CSV file containing workplace locations.            
            pois_csv (str): Path to the CSV file containing points of interest (POI) locations.
            traffic_zone_properties (dict): A dictionary of traffic zone properties with keys:
                'target_size_km' (float): Desired zone size in kilometers.
                'shape' (str): Shape of zones (e.g., "rectangle", "triangle").
                'crop_to_region' (bool): Whether to crop zones to the region of interest.
        """
        print("=========================================")
        print(f"INFO \t Creation of a Region object.")
        print("=========================================")

        self._initialized = False  # Initialization flag

        self.region_geometry = region_geojson
        self.population_density = population_raster
        self.workplaces = workplaces_csv
        self.pois = pois_csv
        self.traffic_zone_properties = traffic_zone_properties

        print(f"INFO \t Successful initialization of input parameters.")

        self._traffic_zones = None # To store resulting traffic zones as a DataFrame

        # Initialize traffic zones immediately
        self._define_traffic_zones()
        self._populate_traffic_zones()

        self._initialized = True  # Initialization flag

    # Traffic zones property (read-only)

    @property
    def traffic_zones(self):
        """pd.DataFrame: Data representing the traffic zones in the region."""
        return self._traffic_zones

    # Properties and Setters with automatic re-population of traffic zones on change

    @property
    def region_geometry(self)-> Polygon:
        """Polygon: The shapely polygon representing the region of interest."""
        return self._region_geometry

    @region_geometry.setter
    def region_geometry(self, path: str):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"The geojson at {path} does not exist.")

        with open(path, 'r') as f:
            geojson_data = json.load(f)

        # Convert the GEOJSON data to a shapely object
        geometry = geojson_data['features'][0]['geometry']
        shapely_shape = shape(geometry)

        self._region_geometry = shapely_shape

        if self._initialized:
            self._populate_traffic_zones()  # Re-populate zones if data has changed and initialized    

    @property
    def population_density(self) -> str:
        """str: The population density data from the raster file."""
        return self._population_density

    @population_density.setter
    def population_density(self, path: str):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"The population density raster at {path} does not exist.")
        
        # Read the raster data using rasterio
        with rasterio.open(path) as src:
            # Get the bounds of the raster
            raster_bounds = src.bounds
            
            # Get the bounds of the region of interest
            region_bounds = self.region_geometry.bounds  # Assuming _region is a shapely Polygon
            
            # Check if the raster covers the region of interest
            if not (raster_bounds[0] <= region_bounds[2] and  # raster left <= region right
                    raster_bounds[2] >= region_bounds[0] and  # raster right >= region left
                    raster_bounds[1] <= region_bounds[3] and  # raster bottom <= region top
                    raster_bounds[3] >= region_bounds[1]):     # raster top >= region bottom
                raise ValueError("The population density raster does not cover the entire region of interest.")
                    
            # If the check passes, assign the path
            self._population_density = path

            if self._initialized:
                self._populate_traffic_zones()  # Re-populate zones if data has changed and initialized   

    @property
    def workplaces(self) -> list:
        """list: A list of tuples representing workplaces (longitude, latitude)"""
        return self._workplaces

    @workplaces.setter
    def workplaces(self, path: str) -> None:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"ERROR \t The CSV for workplaces at {path} does not exist.")
        
        workplaces_center_points = self._load_weighted_locations(path)

        self._workplaces = workplaces_center_points

        if self._initialized:
            self._populate_traffic_zones()  # Re-populate zones if data has changed and initialized 

    @property
    def pois(self) -> list:
        """list: A list of tuples representing points of interest (longitude, latitude)"""
        return self._pois

    @pois.setter
    def pois(self, path: str):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"ERROR \t The CSV for POIs at {path} does not exist.")
        
        pois_center_points = self._load_weighted_locations(path)

        self._pois = pois_center_points

        if self._initialized:
            self._populate_traffic_zones()  # Re-populate zones if data has changed and initialized 

    @property
    def traffic_zone_properties(self) -> dict:
        """dict: A dictionary of traffic zone properties."""
        return self._traffic_zone_properties

    @traffic_zone_properties.setter
    def traffic_zone_properties(self, properties: dict):
        if not isinstance(properties, dict):
            raise ValueError("Traffic zone properties must be a dictionary.")
        
        required_keys = ['target_size_km', 'shape', 'crop_to_region']
        for key in required_keys:
            if key not in properties:
                raise KeyError(f"Missing required traffic zone property: '{key}'")
        
        if properties['target_size_km'] <= 0:
            raise ValueError("Target size must be a positive value.")
        
        self._traffic_zone_properties = properties

        if self._initialized:
            self._define_traffic_zones() # Re-define traffic zones
            self._populate_traffic_zones()  # Re-populate zones if data has changed and initialized

    # Helper Methods

    def _load_weighted_locations(self, path: str) -> list:
        """
        Loads weighted locations from a CSV file and returns a list of tuples representing (longitude, latitude) points.
        
        Args:
            path (str): Path to the CSV file containing locations.
        
        Returns:
            list: A list of tuples with repeated locations based on their weight.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"The CSV at {path} does not exist.")

        center_points = []
        with open(path, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                name = row['name']
                latitude = float(row['latitude'])
                longitude = float(row['longitude'])
                weight = int(row['weight'])

                if weight == 0:
                    print(f"Warning: Skipping {name} row in {path} due to zero weight: {weight}")
                    continue
                if weight < 0:
                    raise ValueError(f"{name} row in {path} contains negative weight")

                # Duplicate the locations according to the weight
                for _ in range(weight):
                    center_points.append((longitude, latitude))

        return center_points

    # Zoning Methods

    def _define_traffic_zones(self):
        """Defines traffic zones based on the specified properties in traffic_zone_properties."""
        print(f"INFO \t Defining traffic zones...")

        shape_type = self._traffic_zone_properties["shape"]
        target_size_km = self._traffic_zone_properties["target_size_km"]
        crop_to_region = self._traffic_zone_properties["crop_to_region"]

        # Create a bbox and zones zoning according to the shape type
        if shape_type == "rectangle":
            self._traffic_zones = self._create_rectangular_zones(target_size_km)
        else:
            raise NotImplementedError(f"Shape type '{shape_type}' is not implemented.")

        # Delete zones outside the boundaries of the shapefile
        if crop_to_region:
            print(f"\t Removing zones outside region of interest (crop to region)...")
            # Identify the rows to remove
            rows_to_remove = self._traffic_zones[self._traffic_zones['is_inside_region'] == False]

            # Drop these rows from the original DataFrame
            self._traffic_zones = self._traffic_zones.drop(rows_to_remove.index)

            print(f"\t > Remaining zones: {len(self._traffic_zones)}")

    def _create_rectangular_zones(self, target_size_km):
        """Creates a bounding box around the region of interest and splits it into rectangles based on the target size"""
        print(f"INFO \t Rectangular zoning...")        
        # Get the bbox and width of the region of interest
        minx, miny, maxx, maxy = self.region_geometry.bounds
        bbox_width_km = geodesic((minx, miny), (maxx, miny)).kilometers
        bbox_height_km = geodesic((minx, miny), (minx, maxy)).kilometers

        # Compute the closest number of integer segments matching the target width 
        n_rectangles = round(bbox_width_km / target_size_km)

        # Calculate the width and height of each zone
        width = (maxx - minx) / n_rectangles
        height = (maxy - miny) / n_rectangles
        
        print(f"\t > Bounding box splitted into {n_rectangles} x {n_rectangles} zones")
        print(f"\t > Zone width: {bbox_width_km/n_rectangles} km")
        print(f"\t > Zone height: {bbox_height_km/n_rectangles} km")         

        grid_data = []

        # Loop to create grid and calculate center of each square        
        for i in range(n_rectangles):     
            for j in range(n_rectangles):
                # ID 
                zone_id = f"{i}_{j}"

                # Bounding box 
                lower_left_x = minx + i * width
                lower_left_y = miny + j * height
                upper_right_x = lower_left_x + width
                upper_right_y = lower_left_y + height

                bbox_geom = box(lower_left_x, lower_left_y, upper_right_x, upper_right_y)                
                bbox_polygon = Polygon(list(bbox_geom.exterior.coords))  # Convert to Polygon                

                # Extract latitude and longitude from the centroid

                # Get the geometric center (centroid) of the bounding box
                center_point = bbox_geom.centroid  # returns a shapely Point

                center_lat = center_point.x  # Longitude
                center_lon = center_point.y  # Latitude

                # Check if the point is within the region of interest
                is_inside = self.region_geometry.contains(center_point)

                # Append everything
                grid_data.append({
                    'id': zone_id, 
                    'centroid': (center_lat, center_lon), 
                    'geometry': bbox_polygon, 
                    'is_inside_region': is_inside
                })

        # Convert to dataframe 
        return pd.DataFrame(grid_data)    

    # Aggregation Methods

    def _populate_traffic_zones(self):
        """Populates traffic zones with aggregated data on population, workplaces, and POIs."""

        print(f"INFO \t Aggregation of geodata into traffic zones...") 

        if self.traffic_zones is None:
            raise ValueError("Traffic zones must be defined before populating them.")

        traffic_zone_data = []

        for _, zone in self.traffic_zones.iterrows():
            zone_geom = zone["geometry"]

            # Population 
            bbox_gdf = gpd.GeoDataFrame({'geometry': [zone_geom]}, crs="EPSG:4326")
            
            population_raster_path = self.population_density # Path to the population raster file
            
            with rasterio.open(population_raster_path) as src: # Read the population raster
                # Clip the raster using the bounding box
                out_image, out_transform = mask(src, [bbox_gdf.geometry.values[0]], crop=True)
                out_meta = src.meta

            n_people = np.sum(out_image[out_image > 0])  # assuming no data values are <= 0  

            # Workplaces
            # Convert the list of center points to shapely Point objects
            points = [Point(lon, lat) for lon, lat in self.workplaces]

            # Count how many points are within the bounding box
            points_within_bbox = [point for point in points if point.within(zone_geom)]
            n_workplaces = len(points_within_bbox)

            # POIs (same logic as for workplaces)
            points = [Point(lon, lat) for lon, lat in self.pois]
            points_within_bbox = [point for point in points if point.within(zone_geom)]
            n_pois = len(points_within_bbox)

            traffic_zone_data.append({
                "id": zone["id"],
                "geometric_center": zone["centroid"],
                "geometry": zone["geometry"],
                "n_people": n_people,
                "n_workplaces": n_workplaces,
                "n_pois": n_pois
            })

        self._traffic_zones = pd.DataFrame(traffic_zone_data) # Update traffic zones

        print(f"\t > Population: {self.traffic_zones["n_people"].sum()}")
        print(f"\t > Workplaces: {self.traffic_zones["n_workplaces"].sum()}")
        print(f"\t > POIs: {self.traffic_zones["n_pois"].sum()}")

    # Region metrics

    def centroid_coords(self) -> tuple:
        """
        Get the coordinates of the centroid of the target area.

        Returns:
            tuple: The (latitude, longitude) coordinates of the centroid.
        """
        centroid = self.region_geometry.centroid
        return centroid.y, centroid.x

    def average_zone_area_km2(self) -> float:
        """
        """
        # Function to calculate the area using geodesic distances
        def calculate_area(polygon):
            # Get the bounds of the polygon
            minx, miny, maxx, maxy = polygon.bounds
            
            # Calculate the area using geodesic
            length = geodesic((minx, miny), (maxx, miny)).kilometers  # Width (in km)
            height = geodesic((minx, miny), (minx, maxy)).kilometers  # Height (in km)
            
            # Area in km²
            area_km2 = length * height
            return area_km2

        areas = self._traffic_zones['geometry'].apply(calculate_area)

        # Calculate the average area
        average_area_km2 = areas.mean()

        return average_area_km2

    # Export and visualization

    def to_map(self, filepath: str) -> None:
        """
        Saves a folium map with simulation setup properties, including region geometry, 
        zone geometry, population, workplaces, POIs.
        """
        df = self.traffic_zones

        # 1. Create an empty map
        m1 = folium.Map(location=self.centroid_coords(), zoom_start=12, tiles='CartoDB Positron', control_scale=True)

        # 2. Add Administrative Boundaries
        def style_function(feature):
            return {
                'color': 'blue',
                'weight': 3,
                'fillColor': 'none',
            }
        
        folium.GeoJson(self.region_geometry, name='Region', style_function=style_function).add_to(m1)

        # 3. Add Simulation bbox
        minx, miny, maxx, maxy = self.region_geometry.bounds
        folium.Rectangle(bounds=[[miny, minx], [maxy, maxx]], fill=True, fill_opacity=0, color='blue', weight=2).add_to(m1)

        # 4. Add rectangles (using apply)
        def add_rectangle(row, colormap, col, map_obj):
            bbox_polygon = row['geometry']
            bbox_coords = bbox_polygon.bounds
            folium.Rectangle(
                bounds=[(bbox_coords[1], bbox_coords[0]), (bbox_coords[3], bbox_coords[2])],
                color=None, fill=True, fill_color=colormap(row[col]), fill_opacity=0.7
            ).add_to(map_obj)

        # Normalize data for destinations and population
        linear1 = cm.LinearColormap(["white", "yellow", "red"], vmin=df['n_workplaces'].min(), vmax=df['n_workplaces'].max())
        linear2 = cm.LinearColormap(["white", "yellow", "red"], vmin=df['n_people'].min(), vmax=df['n_people'].max())
        linear3 = cm.LinearColormap(["white", "yellow", "red"], vmin=df['n_pois'].min(), vmax=df['n_pois'].max())

        # Create FeatureGroups for destinations and population
        workplaces_group = folium.FeatureGroup(name='Number of workplaces', show=False)
        population_group = folium.FeatureGroup(name='Number of people', show=True)
        pois_group = folium.FeatureGroup(name='Number of pois', show=False)

        # Add workplaces rectangles to the group
        df.apply(lambda row: add_rectangle(row, linear1, 'n_workplaces', workplaces_group), axis=1)

        # Add population rectangles to the group
        df.apply(lambda row: add_rectangle(row, linear2, 'n_people', population_group), axis=1)

        # Add pois rectangles to the group
        df.apply(lambda row: add_rectangle(row, linear3, 'n_pois', pois_group), axis=1)

        # Add the FeatureGroups to the map
        workplaces_group .add_to(m1)
        population_group.add_to(m1)
        pois_group.add_to(m1)

        # Add color scales
        linear1.caption = 'Number of workplaces'
        linear1.add_to(m1)
        
        linear2.caption = 'Number of people'
        linear2.add_to(m1)

        linear3.caption = 'Number of POIs'
        linear3.add_to(m1)

        # 6. Add Layer Control
        folium.LayerControl().add_to(m1)
        
        m1.save(filepath)

    def to_csv(self, filepath: str) -> None:
        """
        Saves a csv file with all data
        """
        self.traffic_zones.to_csv(filepath)