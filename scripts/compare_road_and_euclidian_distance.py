# coding: utf-8

"""
A script to calculate and compare Euclidean and road distances between randomly generated coordinates 
within a specified bounding box using OpenRouteService (ORS) API.

## Description:
This script generates a set number of random geographical coordinates within a defined bounding box. 
For each pair of generated coordinates, it calculates:
1. The Euclidean distance using the geodesic method from the geopy library.
2. The road distance using the OpenRouteService API.

The results are stored in a Pandas DataFrame, which is then exported to a CSV file for further analysis.

## Usage:
1. Replace `ors_api_key` with your own OpenRouteService API key to enable access to the routing service.
2. Adjust the bounding box coordinates (`lat_min`, `lon_min`, `lat_max`, `lon_max`) to set the area of interest.
3. Set the `num_points` variable to determine how many random locations to generate.
4. Run the script. The results will be saved in a CSV file named `road_vs_euclidian_distances.csv`.

## Dependencies:
- openrouteservice: To interact with the OpenRouteService API for routing distances.
- geopy: To calculate Euclidean distances between coordinates.
- pandas: For data manipulation and CSV file creation.

## Notes:
- The script includes error handling for rate limits and API errors when fetching road distances.
- There is a delay of 1.5 seconds between API requests to avoid exceeding the rate limit.
"""

import openrouteservice
from geopy.distance import geodesic
import pandas as pd
from random import uniform
import time

# Initialize ORS client with your API key
ors_api_key = '5b3ce3597851110001cf6248879c0a16f2754562898e0826e061a1a3'
client = openrouteservice.Client(key=ors_api_key)

# Bounding box for the area of interest (latitude and longitude)
lat_min, lon_min = 38.6394, 8.8335
lat_max, lon_max = 38.9062, 9.0982

# Generate random coordinates
def generate_random_coordinates(num_points, lat_min, lat_max, lon_min, lon_max):
    return [(uniform(lat_min, lat_max), uniform(lon_min, lon_max)) for _ in range(num_points)]

# Calculate Euclidean distance using geopy
def euclidean_distance(coord1, coord2):
    return geodesic(coord1, coord2).meters

# Function to get road distance from ORS
def get_road_distance(coord1, coord2):
    try:
        routes = client.directions(
            coordinates=[coord1, coord2],
            profile='driving-car',
            format='geojson',
            radiuses = [3000, 3000]
        )

        time.sleep(1.5)
        return routes['features'][0]['properties']['segments'][0]['distance']

    except openrouteservice.exceptions._OverQueryLimit as e:
        print("Rate limit exceeded, sleeping...")
        time.sleep(60)  # Wait for 60 seconds before retrying
    except (IndexError, KeyError, openrouteservice.exceptions.ApiError) as e:
        print(f"Error fetching road distance: {e}")
        return None

# Number of random points
num_points = 16  # Adjust as needed

# Generate random locations
locations = generate_random_coordinates(num_points, lat_min, lat_max, lon_min, lon_max)

# Prepare DataFrame
results = []

# Calculate distances for all pairs
k = 0
for i in range(num_points):
    for j in range(i + 1, num_points):
        k = k + 1

        print(f"Calculating point {k} out of {num_points*(num_points-1)/2}")
        coord1 = locations[i]
        coord2 = locations[j]

        euclid_dist = euclidean_distance(coord1, coord2)
        road_dist = get_road_distance(coord1, coord2)

        results.append({
            'coord1': coord1,
            'coord2': coord2,
            'euclidean_distance': euclid_dist,
            'road_distance': road_dist
        })

# Create DataFrame
df = pd.DataFrame(results)

# Save results to CSV
df.to_csv('road_vs_euclidian_distances.csv', index=False)

print("Distance calculations completed and saved to 'distances.csv'.")