# coding: utf-8

""" 
A python script to simulate the spatio-temporal charging demand based on 
mobility simulation.
"""

import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import LineString, Point, box, Polygon
from shapely import wkt
from shapely.ops import transform
from rasterio.features import geometry_mask
from rasterio.features import shapes
from shapely.geometry import mapping
import pyproj
from pyproj import Transformer
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from branca.colormap import LinearColormap
import numpy as np
import math
import os
import folium
from folium.plugins import AntPath
from folium.plugins import HeatMap
import pickle


from dotenv import load_dotenv
from pathlib import Path
import osmnx as ox
import branca.colormap as cm

from evpv.mobilitysim import MobilitySim
from evpv.chargingdemand import ChargingDemand
from evpv import helpers as hlp

#############################################
# PARAMETERS - MODIFY ACCORDING TO YOUR NEEDS
#############################################

"""
Environment variables
"""

load_dotenv() # take environment variables from .env

INPUT_PATH = Path( str(os.getenv("INPUT_PATH")) )
OUTPUT_PATH = Path( str(os.getenv("OUTPUT_PATH")) )

"""
Global parameters 
"""

shapefile_path = INPUT_PATH / "gadm41_ETH_1.json" # Addis Ababa administrative boundaries
population_density_path = INPUT_PATH / "GHS_POP_merged_4326_3ss_V1_0_R8andR9_C22.tif" # Population density raster

commuting_zone = {
            "isochrone_center": (38.74, 9.02),            
            "max_time_min": 10,
            "isochrone_timestep_min": 10
        }

n_subdivisions = 8 # Number of subdivisions of the bbox to create traffic analysis zones
road_network_filter_string = '["highway"~"^(primary|secondary|tertiary)"]' # Roads used in the road network
workplaces_tags = { # Tags used to get workplaces
            "building": ["industrial", "office"],
            "company": [],
            "landuse": ["industrial"],
            "industrial": [],
            "office": ["company", "government"],
            "amenity": ["university", "research_institute", "conference_centre", "bank", "hospital", "townhall", "police", "fire_station", "post_office", "post_depot"]
        }

share_active = 0.1
share_unemployed = 0.227
share_home_office = 0.0
mode_share = 1.0
vehicle_occupancy = 1.2

model = "gravity_exp_01"
attraction_feature = "population"
cost_feature = "distance_centroid"
taz_center = "centroid"

use_cached_data = True

#############################################
### MOBILITY SIMULATION (home-work-home) ####
#############################################

mobsim = None # Init the mobsim object for the mobility simulation 

unique_id = hlp.create_unique_id([shapefile_path, population_density_path, n_subdivisions, road_network_filter_string, workplaces_tags, share_active, share_unemployed, share_home_office, mode_share, vehicle_occupancy, model, attraction_feature, cost_feature, taz_center]) # Unique ID from input variables - ensures that we redo the simulation
pickle_filename = OUTPUT_PATH / f"evpv_Tmp_MobilitySim_Cache_{unique_id}.pkl" # Unique pickle filename usinb 


# If True, try to use cached pickle object
if use_cached_data and os.path.isfile(pickle_filename): 
    print("INFO \t Mobility simulation: loading object from pickle file...")
    with open(pickle_filename, 'rb') as file:
        mobsim = pickle.load(file)

else:
    # 1. MobilitySim object initialization 

    mobsim = MobilitySim(
        target_area_shapefile = shapefile_path,
        population_density = population_density_path, 
        commuting_zone = commuting_zone, 
        n_subdivisions = n_subdivisions,
        road_network_filter_string = road_network_filter_string,
        workplaces_tags = workplaces_tags)

    # 2. Trip generation from statistics

    mobsim.trip_generation(
        share_active = share_active, 
        share_unemployed = share_unemployed, 
        share_home_office = share_home_office, 
        mode_share = mode_share, 
        vehicle_occupancy = vehicle_occupancy
    )

    # 3. Trip ditribution using SIM

    mobsim.trip_distribution(model = model, attraction_feature = attraction_feature, cost_feature = cost_feature, taz_center = taz_center)

    # 4. Storing data
    print("INFO \t Saving MobilitySim object to pickle file")
    with open(pickle_filename, 'wb') as file:
        pickle.dump(mobsim, file)

# 4. Storing outputs

mobsim.flows.to_csv(OUTPUT_PATH / "evpv_Result_MobilitySim_OriginDestinationFlows.csv", index=False) # Store aggregated TAZ features as csv
mobsim.traffic_zones.to_csv(OUTPUT_PATH / "evpv_Result_MobilitySim_TrafficAnalysisZones.csv", index=False) # Store aggregated TAZ features as csv

#############################################
############### CHARGING NEEDS ##############
#############################################

chargedem = ChargingDemand(
    mobsim = mobsim, 
    ev_consumption = 0.2,
    charging_efficiency = 0.9)

time, power_profile, max_cars_plugged_in = chargedem.load_profile(mean_arrival_time = 18, std_arrival_time = 0.1, charging_power = 3)

chargedem.taz_properties.to_csv(OUTPUT_PATH / "evpv_Result_ChargingDemand_TAZProperties.csv", index=False) # Store aggregated TAZ features as csv

#############################################
################ VISUALISATION ##############
#############################################
print(f"Maximum number of cars plugged in at the same time: {max_cars_plugged_in}")

plt.figure(figsize=(10, 6))
plt.plot(time, power_profile, label='Power Demand (MWh)')
plt.xlabel('Time (hours)')
plt.ylabel('Power Demand (MWh)')
plt.title('EV Charging Power Profile')
plt.legend()
plt.grid(True)
plt.show()



#############################################
############### OTHER ANALYSIS ##############
#############################################

################## Routing ##################

# mobsim.allocate_routes() # Do not comment
# df = mobsim.flows

# m = folium.Map(location=mobsim.centroid_coords, zoom_start=12, tiles='CartoDB Positron', control_scale=True) # Create the map

# # Normalize flow values for color scaling
# min_flow = df['Flow'].min()
# max_flow = df['Flow'].max()
# df['normalized_flow'] = (df['Flow'] - min_flow) / (max_flow - min_flow)

# # Function to get color based on normalized flow
# def get_color(normalized_flow):
#     return f'#{int(255*normalized_flow):02x}00{int(255*(1-normalized_flow)):02x}'

# # Function to get line width based on normalized flow
# def get_width(normalized_flow):
#     return 1 + normalized_flow * 9  # Width ranges from 1 to 10

# # Add routes to the map
# for idx, row in df.iterrows():
#     geojson = gpd.GeoSeries([row['Geometry']]).__geo_interface__
#     color = get_color(row['normalized_flow'])
#     width = get_width(row['normalized_flow'])
    
#     folium.GeoJson(
#         geojson,
#         style_function=lambda x, color=color, width=width: {
#             'color': color,
#             'weight': width,
#             'opacity': 0.7
#         }
#     ).add_to(m)

# # Add a color scale legend
# colormap = folium.LinearColormap(colors=['blue', 'red'], vmin=min_flow, vmax=max_flow)
# colormap.add_to(m)
# colormap.caption = 'Flow'

# # Save the map to an HTML file
# m.save(OUTPUT_PATH / "traffic_flows_map.html")

#############################################








# Method 1
# # Function to get color based on normalized flow
# def get_color(normalized_flow):
#     return f'#{int(255*normalized_flow):02x}00{int(255*(1-normalized_flow)):02x}'

# # Add routes to the map
# for idx, row in df.iterrows():
#     geojson = gpd.GeoSeries([row['Geometry']]).__geo_interface__
#     color = get_color(row['normalized_flow'])
    
#     folium.GeoJson(
#         geojson,
#         style_function=lambda x, color=color: {
#             'color': color,
#             'weight': 2,
#             'opacity': 0.7
#         }
#     ).add_to(m)

# # Add a color scale legend
# colormap = folium.LinearColormap(colors=['blue', 'red'], vmin=min_flow, vmax=max_flow)
# colormap.add_to(m)
# colormap.caption = 'Flow'



# Method 2



# # Function to get color based on normalized flow
# def get_color(normalized_flow):
#     return f'#{int(255*normalized_flow):02x}00{int(255*(1-normalized_flow)):02x}'

# # Function to get line width based on normalized flow
# def get_width(normalized_flow):
#     return 1 + normalized_flow * 9  # Width ranges from 1 to 10

# # Add routes to the map
# for idx, row in df.iterrows():
#     geojson = gpd.GeoSeries([row['Geometry']]).__geo_interface__
#     color = get_color(row['normalized_flow'])
#     width = get_width(row['normalized_flow'])
    
#     folium.GeoJson(
#         geojson,
#         style_function=lambda x, color=color, width=width: {
#             'color': color,
#             'weight': width,
#             'opacity': 0.7
#         }
#     ).add_to(m)

# # Add a color scale legend
# colormap = folium.LinearColormap(colors=['blue', 'red'], vmin=min_flow, vmax=max_flow)
# colormap.add_to(m)
# colormap.caption = 'Flow'



# Rasterize the result

# # Step 2: Load the reference population raster
# reference_raster_path = OUTPUT_PATH / "population_density_cropped.tiff"
# with rasterio.open(reference_raster_path) as src:
#     meta = src.meta.copy()
#     transform = src.transform
#     out_shape = src.shape

# # Step 3: Create an empty raster with the same metadata
# output_raster = np.zeros(out_shape, dtype=np.float32)

# # Step 4: Rasterize the LineStrings while accumulating flow values
# for idx, row in df.iterrows():
#     geom = row['Geometry']
#     flow = row['Flow']
    
#     # Create a generator of pixel-wise shapes from the LineString
#     shapes_gen = shapes(geom, transform=transform)
    
#     # Iterate over each shape and accumulate flow values in the output raster
#     for geom, _ in shapes_gen:
#         # Convert the geometry to a Shapely shape
#         geom_shape = shape(geom)
        
#         # Calculate the pixel indices that intersect with the shape
#         minx, miny, maxx, maxy = geom_shape.bounds
#         col_min, row_min = src.index(minx, miny)
#         col_max, row_max = src.index(maxx, maxy)
        
#         # Iterate over the intersecting pixels and accumulate flow values
#         for row in range(row_min, row_max + 1):
#             for col in range(col_min, col_max + 1):
#                 if geom_shape.intersects(shape(src.pixel_bbox(row, col))):
#                     output_raster[row, col] += flow

# # Step 5: Save the resulting raster
# output_raster_path = 'path_to_output_flow_raster.tif'
# with rasterio.open(output_raster_path, 'w', **meta) as dst:
#     dst.write(output_raster, 1)

# print(f"Output raster saved to {output_raster_path}")

# # Function to calculate weight (flow intensity) for each edge
# def calculate_edge_weight(edge_flows, edge):
#     flow = edge_flows.get(edge, 0)  # Get flow value for edge, default to 0 if not present
#     return flow

# # Create a list of edges with flow intensity
# edges_with_flow = [(edge[0], edge[1], calculate_edge_weight(edge_flows, edge)) for edge in G.edges()]

# # Extract coordinates of nodes
# node_positions = {node: (data['y'], data['x']) for node, data in G.nodes(data=True)}

# # Create a list of edge segments with flow intensity and coordinates
# heat_data = []
# for u, v, flow in edges_with_flow:
#     if u in node_positions and v in node_positions:
#         heat_data.append([node_positions[u], node_positions[v], flow])

# # Convert coordinates to float and ensure they are in (latitude, longitude) format
# heat_data_float = []
# for item in heat_data:
#     if len(item) == 3:  # Check if item has three values (coordinates and flow)
#         start_coord = tuple(item[0])  # Reverse and convert to tuple
#         end_coord = tuple(item[1])  # Reverse and convert to tuple
#         weight = item[2]  # Flow intensity
#         heat_data_float.append((start_coord[0], start_coord[1], weight))  # Ensure each coordinate pair is (lat, lon, weight)
#         heat_data_float.append((end_coord[0], end_coord[1], weight))  # Add both start and end points separately
#     elif len(item) == 2:  # Handle case where item has two values (coordinates and flow)
#         start_coord = tuple(item[0])  # Reverse and convert to tuple
#         end_coord = tuple(item[1])  # Reverse and convert to tuple
#         weight = item[1]  # Flow intensity
#         heat_data_float.append((start_coord[0], start_coord[1], weight))  # Ensure each coordinate pair is (lat, lon, weight)
#         heat_data_float.append((end_coord[0], end_coord[1], weight))  # Add both start and end points separately

# # Create a HeatMap layer
# heat_map = HeatMap(heat_data_float, min_opacity=0.5, max_val=max(edge_flows.values()), radius=15, blur=10)

# # Add HeatMap layer to the map
# heat_map.add_to(m)




# Function to get the coordinates of an edge
# def get_edge_coords(G, edge):
#     if len(edge) == 3:
#         u, v, key = edge
#     else:
#         u, v = edge
#     u_coords = (G.nodes[u]['y'], G.nodes[u]['x'])
#     v_coords = (G.nodes[v]['y'], G.nodes[v]['x'])
#     return [u_coords, v_coords]

# # Create a MarkerCluster to aggregate flows based on zoom level
# marker_cluster = folium.plugins.MarkerCluster().add_to(m)

# # Determine min and max flow values for normalization
# min_flow = min(edge_flows.values())
# max_flow = max(edge_flows.values())

# # Function to calculate color based on flow intensity
# def calculate_color(flow):
#     normalized_flow = (flow - min_flow) / (max_flow - min_flow)
#     return folium.LinearColormap(['blue', 'red'], vmin=0, vmax=1).to_step(10)(normalized_flow)

# # Function to aggregate flows of adjacent edges
# def aggregate_adjacent_flows(G, edge_flows):
#     aggregated_edge_flows = {}
#     for edge, flow in edge_flows.items():
#         u, v = edge[:2]  # For simplicity, consider only u-v for undirected or first segment in multigraph
#         if (u, v) in aggregated_edge_flows:
#             aggregated_edge_flows[(u, v)] += flow
#         else:
#             aggregated_edge_flows[(u, v)] = flow
#     return aggregated_edge_flows

# # Aggregate flows of adjacent edges
# aggregated_edge_flows = aggregate_adjacent_flows(G, edge_flows)


# # Create a feature group for edges
# edge_group = folium.FeatureGroup(name='Edges').add_to(m)

# # Add aggregated edges to the map with flow visualization
# for edge, flow in aggregated_edge_flows.items():
#     coords = get_edge_coords(G, edge)
#     color = calculate_color(flow)
#     popup_text = f"Aggregated Flow: {flow}"  # Example popup text
#     folium.PolyLine(coords, color=color, weight=5, opacity=0.7, popup=popup_text).add_to(edge_group)

# # Add LayerControl to toggle edge visibility
# folium.LayerControl().add_to(m)


# # Add edges to the map with flow visualization
# for edge, flow in edge_flows.items():
#     coords = get_edge_coords(G, edge)
#     color = mcolors.to_hex(cmap(norm(flow)))
#     folium.PolyLine(coords, color=color, weight=2, opacity=0.7).add_to(m)


# # Add colormap legend to the map
# colormap = LinearColormap(colors=['blue', 'red'], vmin=min(edge_flows.values()), vmax=max(edge_flows.values()))
# colormap.add_to(m)



# Save the map to an HTML file
#m.save(OUTPUT_PATH / "traffic_flows_map.html")

# Workplaces as a function of population

"""
##################### MobilitySim simulation #####################
"""

"""
##################### MobilitySim simulation #####################
"""

"""
##################### MobilitySim simulation #####################
"""











# # 1. Creating the object 

# mobsim = MobilitySim(
#     target_area_shapefile = shapefile_path,
#     population_density = population_density_path, 
#     commuting_zone = commuting_zone, 
#     n_subdivisions = n_subdivisions,
#     road_network_filter_string = road_network_filter_string,
#     workplaces_tags = workplaces_tags)

# mobsim.traffic_zones.to_csv(OUTPUT_PATH / "traffic_zones.csv", index=False)

# # 2. Pre-analysis: number of workplaces as a function of the population
# #Is the population a good proxy for workplaces?

# # plt.figure(figsize=(10, 6))
# # plt.scatter(mobsim.traffic_zones['population'], mobsim.traffic_zones['workplaces'], color='blue')

# # # Adding title and labels
# # plt.title('Number of Workplaces as a Function of Population')
# # plt.xlabel('Population')
# # plt.ylabel('Number of Workplaces')

# # plt.show()
# # plt.savefig(OUTPUT_PATH / 'population_vs_workplaces.png')

# # 3. Create a folium map and add all the data

# mymap = folium.Map(location=mobsim.centroid_coords, zoom_start=12, tiles='CartoDB Positron', control_scale=True) # Create the map


# ##### 3.1 Administrative boundaries

# # Define style function to only show lines
# def style_function(feature):
#     return {
#         'color': 'blue',  # Set line color
#         'weight': 3,      # Set line weight
#         'fillColor': 'none',  # Set fill color to 'none'
#     }

# folium.GeoJson(mobsim.target_area_shapefile['features'][0]['geometry'], name='Administrative boundary', style_function=style_function).add_to(mymap)

# ##### 3.2 Simulation bbox

# minx, miny, maxx, maxy = mobsim.simulation_bbox

# # Create a rectangle using the bounding box coordinates
# rectangle = folium.Rectangle(
#     bounds=[[miny, minx], [maxy, maxx]],
#     fill=True,  # Fill the rectangle
#     fill_opacity=0,  # Set the opacity of the fill color
#     color='blue',  # Border color
#     weight=2,  # Border width
# )

# # Add the rectangle to the map
# rectangle.add_to(mymap)

# ##### 3.3 Population data

# mymap = hlp.add_raster_to_folium(mobsim.population_density, mymap)

# ##### 3.4 Road network

# # # Convert the network to a GeoDataFrame
# # gdf = ox.graph_to_gdfs(mobsim.road_network, nodes=False, edges=True)

# # # Add the road network to the map /!\ Heavy process
# # folium.GeoJson(gdf, name='Road Network', style_function=lambda x:{'fillColor': '#000000', 'color': '#000000'}).add_to(mymap)

# ##### 3.5 Workplaces

# #Add workplaces

# # for point in mobsim.workplaces:
# #     folium.Marker(location=[point[1], point[0]], popup="Center point").add_to(mymap)

# ##### 3.6 Markers for the nearest nodes

# # for idx, row in mobsim.traffic_zones.iterrows():
# #     #nearest_node_lat, nearest_node_lon = row['nearest_node']
# #     nearest_node_lat, nearest_node_lon = row['geometric_center']
# #     folium.Marker(
# #         location=[nearest_node_lon, nearest_node_lat],
# #         icon=folium.Icon(color='red'),
# #         popup=f"{nearest_node_lat}, {nearest_node_lon} - Pop: {int(row['population'])} - Work: {int(row['workplaces'])}"
# #     ).add_to(mymap)

# ##### 3.7 TAZs

# # Function to add rectangles to the map
# def add_rectangle(row):
#     # Parse the WKT string to create a Polygon object
#     bbox_polygon = row['bbox']
#     bbox_coords = bbox_polygon.bounds
    
#     # Add rectangle to map
#     folium.Rectangle(
#         bounds=[(bbox_coords[1], bbox_coords[0]), (bbox_coords[3], bbox_coords[2])],
#         color='grey',
#         fill=True,
#         fill_color='grey',
#         fill_opacity=0.0
#     ).add_to(mymap)

# # # Apply the function to each row in the DataFrame
# mobsim.traffic_zones.apply(add_rectangle, axis=1)

# # # Display the map
# folium.LayerControl().add_to(mymap)
# mymap.save(OUTPUT_PATH / "data_map.html")


# """
# ##################### Trip generation #####################
# """

# # Generation of number of home-work-home and home-study-home trips by car 

# mobsim.trip_generation(
#     share_active = 0.76, 
#     share_unemployed = 0.227, 
#     share_home_office = 0.0, 
#     mode_share = 1.0, 
#     vehicle_occupancy = 1.0
# )

# # Creating a map with the number of commuters

# df = mobsim.traffic_zones

# m = folium.Map(location=mobsim.centroid_coords, zoom_start=12, tiles='CartoDB Positron') # Create the map

# # Normalize population data for color scaling
# linear = cm.LinearColormap(["green", "yellow", "red"], vmin=df['n_commuters'].min(), vmax=df['n_commuters'].max())

# # Add polygons to the map
# for idx, row in df.iterrows():

#     bbox_polygon = row['bbox']
#     bbox_coords = bbox_polygon.bounds

#     folium.Rectangle(
#         bounds=[(bbox_coords[1], bbox_coords[0]), (bbox_coords[3], bbox_coords[2])],
#         color=None,
#         fill=True,
#         fill_color=linear(row.n_commuters),
#         fill_opacity=0.7,
#         tooltip=f'Commuters: {row.n_commuters} - Car trips: {row.n_trips} '
#     ).add_to(m)

# # Display the map
# m.save(OUTPUT_PATH / 'n_commuters.html')

# """
# ##################### Trip distribution #####################
# """

# # mobsim.trip_distribution(model = "gravity_power_1", attraction_feature = "population", cost_feature = "distance_centroid", taz_center = "centroid")

# # df = mobsim.flows

# # df.to_csv(OUTPUT_PATH / "mobility_flows.csv", index=False)

# # flow_weighted_avg_distance = np.average(df['Travel Distance (km)'], weights=df['Flow'])
# # flow_weighted_avg_centroid = np.average(df['Centroid Distance (km)'], weights=df['Flow'])

# # print("Flow-weighted average travel distance:", flow_weighted_avg_distance)
# # print("Flow-weighted average travel distance btwn centroids:", flow_weighted_avg_centroid)

# # print("Total flow:", df['Flow'].sum())
# # print("Total trips:", mobsim.traffic_zones['n_trips'].sum())


# # mobsim.trip_distribution(model = "gravity_exp_01", attraction_feature = "population", cost_feature = "distance_centroid", taz_center = "centroid")
# # df = mobsim.flows
# # flow_weighted_avg_centroid = np.average(df['Centroid Distance (km)'], weights=df['Flow'])
# # print("Flow-weighted average travel distance btwn centroids exp:", flow_weighted_avg_centroid)

# mobsim.trip_distribution(model = "gravity_exp_01", attraction_feature = "workplaces", cost_feature = "time_road", taz_center = "centroid")

# df = mobsim.flows
# flow_weighted_avg_centroid = np.average(df['Centroid Distance (km)'], weights=df['Flow'])
# flow_weighted_avg_distance = np.average(df['Travel Distance (km)'], weights=df['Flow'])

# print("Flow-weighted average travel distance:", flow_weighted_avg_distance)
# print("Flow-weighted average travel distance btwn centroids:", flow_weighted_avg_centroid)



# #######################################
# # Plot the outflows from an origin on a map
# #######################################

# df2 = mobsim.traffic_zones

# # Filter df1 for a specific origin
# origin = '6_6'
# filtered_df1 = df[df['Origin'] == origin]

# m1 = folium.Map(location=mobsim.centroid_coords, zoom_start=12, tiles='CartoDB Positron', control_scale=True) # Create the map

# # Normalize population data for color scaling
# linear = cm.LinearColormap(["green", "yellow", "red"], vmin=filtered_df1['Flow'].min(), vmax=filtered_df1['Flow'].max())

# # Add polygons to the map
# for idx, row in filtered_df1.iterrows():

#     bbox_polygon = df2[df2['id'] == row.Destination]['bbox'].values[0]
#     bbox_coords = bbox_polygon.bounds

#     folium.Rectangle(
#         bounds=[(bbox_coords[1], bbox_coords[0]), (bbox_coords[3], bbox_coords[2])],
#         color=None,
#         fill=True,
#         fill_color=linear(row.Flow),
#         fill_opacity=0.7,
#         tooltip=f'Commuters: {row.Flow} - Car trips: {row.Flow} '
#     ).add_to(m1)

# # Display the map
# m1.save(OUTPUT_PATH / 'a.html')


# print("Flow-weighted average travel time:", flow_weighted_avg_time)

# df = mobsim.flows_car
# print(mobsim.flows_car)

# # Calculate flow-weighted average travel time and distance
# flow_weighted_avg_time = np.average(df['Travel Time (min)'], weights=df['Flow'])
# flow_weighted_avg_distance = np.average(df['Travel Distance (km)'], weights=df['Flow'])

# print("Flow-weighted average travel time:", flow_weighted_avg_time)
# print("Flow-weighted average travel distance:", flow_weighted_avg_distance)

#Plot histogram of total flow for each bin of travel time
# plt.hist(df['Centroid Distance (km)'], bins=200, weights=df['Flow'], color='blue', edgecolor='black')
# plt.xlabel('Centroid Distance (km)')
# plt.ylabel('Total Flow')
# plt.title('Total Flow as a function of Centroid Distance (km)')
# plt.grid(True)
# plt.show()











# df.to_csv('flow_distanc.csv', index=False) 

# Expand coordinates_df into separate latitude and longitude columns
# mobsim.traffic_zones[['longitude', 'latitude']] = pd.DataFrame(mobsim.traffic_zones['geometric_center'].tolist(), index=mobsim.traffic_zones.index) 

# # Merge flows_df with coordinates_df to get coordinates for origins and destinations
# merged_df = mobsim.flows_car.merge(mobsim.traffic_zones, left_on='Origin', right_on='id').rename(columns={'latitude': 'origin_lat', 'longitude': 'origin_lon'})
# merged_df = merged_df.merge(mobsim.traffic_zones, left_on='Destination', right_on='id').rename(columns={'latitude': 'dest_lat', 'longitude': 'dest_lon'})

# # Create a folium map centered around the average coordinates
# mymap = folium.Map(location=[merged_df['origin_lat'].mean(), merged_df['origin_lon'].mean()], zoom_start=13)

# # Apply the function to each row in the DataFrame
# mobsim.traffic_zones.apply(add_rectangle, axis=1)

# # Function to add lines to the map
# # Function to add lines to the map
# def add_flow_line(row):
#     if row['Flow'] > 1000:  # Only add flows with non-zero value
#         origin = (row['origin_lat'], row['origin_lon'])
#         destination = (row['dest_lat'], row['dest_lon'])
#         # Create polyline with arrows at the end
#         folium.plugins.AntPath(
#             locations=[origin, destination],
#             color='red',
#             use_arrows=True,  # Display arrows at the end
#             delay=1000000000,  # Delay between each arrow
#             dash_array=[10, 20],  # Dash pattern to make the arrow part dashed
#             weight=20,
#             opacity=0.6
#         ).add_to(mymap)


# # Apply the function to each row in the DataFrame
# merged_df.apply(add_flow_line, axis=1)



# Save the map to an HTML file
# mymap.save('flow_map.html')


