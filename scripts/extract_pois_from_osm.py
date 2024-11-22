# coding: utf-8

"""
This script retrieves Points of Interest (POIs) from OpenStreetMap (OSM) using the OSMNx library.
It divides a large bounding box into smaller segments to circumvent limitations on data retrieval size.
The retrieved POIs are stored as both GeoJSON and CSV files.

## Usage
1. **Set Parameters**: Adjust the bounding box coordinates and the tags for POIs as needed. You can also modify the number of rows and columns for the grid that splits the bounding box.
2. **Run the Script**: Execute the script to fetch POIs. The script will print progress messages to the console.
3. **Output**: Upon completion, two files will be generated:
   - `center_points.geojson`: A GeoJSON file containing the center coordinates of the fetched POIs.
   - `center_points.csv`: A CSV file listing unique coordinates along with a count of occurrences.

## Important Notes
- The behavior of OSMNx can be unpredictable. When querying a large area, the library may return fewer POIs than expected without warning. To mitigate this, the script splits the area into smaller bounding boxes and continues to fetch data until the total number of retrieved POIs stabilizes.
- If you experience inconsistent results, consider further subdividing the bounding box until the POI count remains constant.

Dependencies:
- geopandas
- pandas
- numpy
- shapely
- osmnx

Make sure to install these packages using pip or conda before running the script.
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import math
import time
from shapely.geometry import Point

import osmnx as ox

#################### PARAMETERS ####################
 
# No cache and no log
ox.config(use_cache=False, log_console=False)

# Bounding box (https://bboxfinder.com/)
west, south, east, north = 38.639904,8.8331149,38.9080529,9.0985761 # Addis (38.639904,8.8331149,38.9080529,9.0985761)

# tags = {
#     "building": ["industrial", "office"],
#     "company": True,
#     "landuse": ["industrial", "commercial", "retail"],
#     "industrial": True,
#     "office": True,
#     "amenity": [
#         "university",             # Large universities often have many staff members
#         "research_institute",      # Research institutes tend to have a substantial workforce
#         "hospital",                # Hospitals have a high number of employees
#         "townhall",                # Municipal buildings with a large staff
#         "conference_centre",       # May employ a large number of service staff
#         "factory",                 # Industrial sites with large staff numbers
#         "corporate_office",        # Corporate offices tend to have large workforces
#         "government",              # Government offices
#         "bank",                    # Larger banks can have many employees
#         "police",                  # Police stations with a substantial workforce
#         "fire_station",            # Fire stations in large cities may have multiple shifts
#         "post_office",             # Post offices can have a significant number of employees
#         "call_centre",             # High-density work environments
#         "logistics_centre"         # Warehouses and logistics centers
#     ]
# }

tags = {
    "amenity": [
        # Transport-related amenities
        "fuel", "parking", "parking_entrance", "bicycle_parking",
        
        # Education-related amenities
        "college", "university", "school", "kindergarten", "library", "music_school", "language_school",
        
        # Health-related amenities
        "clinic", "dentist", "doctors", "hospital", "pharmacy", "veterinary",
        
        # Food and drink
        "cafe", "ice_cream", "internet_cafe", "restaurant", "fast_food", "bar", "pub", "biergarten",
        
        # Entertainment and leisure
        "theatre", "cinema", "music_venue", "nightclub", "casino", "gambling", "stripclub",
        
        # Cultural and community centers
        "arts_centre", "community_centre", "social_centre", "exhibition_centre",        
    
        # Tourism-related amenities
        "attraction", "viewpoint", "aquarium", "beach_resort", "gallery", "museum", "theme_park", "zoo", "artwork"
    ],
    "shop": [
        "supermarket", "mall", "department_store", "convenience"
    ],
    "tourism": [
        "hotel", "guest_house", "hostel", "motel", "camp_site", "apartment"
    ],
    "leisure": [
        "stadium", "sports_centre", "swimming_pool", "fitness_centre"
    ]
}

# Define the number of rows and columns for the grid
n_rows, n_cols = 1, 1  # Adjust these values to control the size of the chunks

print(f"INFO \t Getting the destinations from OSM. Tags: {tags}")

#################### GETTING OSM POIS USING OSMNX AND SPLITTING THE AREA #################### 

# Calculate the step sizes
lat_step = (north - south) / n_rows
lon_step = (east - west) / n_cols

# List to store all bounding boxes
bbox_list = []

# Generate the bounding boxes by splitting the initial bounding box
for i in range(n_rows):
    for j in range(n_cols):
        # Calculate the coordinates for each bounding box
        south_i = south + i * lat_step
        north_i = south + (i + 1) * lat_step
        west_j = west + j * lon_step
        east_j = west + (j + 1) * lon_step

        # Add the bounding box to the list
        bbox_list.append([north_i, south_i, east_j, west_j])

# Initialize a list to store all POIs
all_pois = []

center_points = []

# Iterate over each bounding box and fetch the data
for idx, bbox in enumerate(bbox_list):
    print(f"Fetching data for bbox {idx+1}/{len(bbox_list)}: {bbox}")
    
    try:
        # Fetch the data for the current bounding box
        pois = ox.features.features_from_bbox(bbox=bbox, tags=tags)        
       	all_pois.append(pois)

       	# Convert ways into nodes and get the coordinates of the center point - Do not store the relations
        for index, row in pois.iterrows():
        	if index[0] == 'way':
        		shapefile = row['geometry']
        		center_point = shapefile.centroid
        		center_points.append((center_point.x, center_point.y))
        	if index[0] == 'node':
        		center_points.append((row['geometry'].x, row['geometry'].y))
        	else:
        		break

        print(f"Completed fetching data for bbox {bbox}.")
        time.sleep(1)  # Optional: Add delay to avoid overloading the server
    except Exception as e:
        print(f"Error retrieving data for bbox {bbox}: {e}")


print(f"Number of POIs: {len(center_points)}")

#################### Store POIS as GeoJSON #################### 

# Convert the list of coordinates to a DataFrame
df = pd.DataFrame(center_points, columns=['longitude', 'latitude'])

# Create a GeoDataFrame from the DataFrame
# Convert longitude and latitude to Point geometries
geometry = [Point(lon, lat) for lon, lat in zip(df['longitude'], df['latitude'])]
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

# Optional: Add a unique ID column
gdf['name'] = [f"id_{i+1}" for i in range(len(gdf))]

# Save the GeoDataFrame to a GeoJSON file
gdf.to_file("center_points.geojson", driver="GeoJSON")

#################### Store POIS as csv #################### 

# Create a dictionary to store the count of each unique coordinate
coord_dict = {}

# Count occurrences of each coordinate to reduce file size
for lon, lat in center_points:
    if (lon, lat) in coord_dict:
        coord_dict[(lon, lat)] += 1
    else:
        coord_dict[(lon, lat)] = 1

# Prepare data for CSV with unique IDs
csv_data = [{"name": f"id_{index+1}", "latitude": lat, "longitude": lon, "weight": weight}
            for index, ((lon, lat), weight) in enumerate(coord_dict.items())]

# Create a DataFrame
df = pd.DataFrame(csv_data)

# Save the DataFrame to a CSV file
df.to_csv("center_points.csv", index=False)

print("Data saved")


