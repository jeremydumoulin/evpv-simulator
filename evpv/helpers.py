# coding: utf-8

""" 
Some useful generic functions for gtfs4ev classes.
"""

import pandas as pd
from shapely.geometry import LineString, Point, shape, Polygon, box
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.mask import mask
from rasterio.plot import reshape_as_image
from pyproj import Geod
from PIL import Image
import tempfile
import folium
import branca.colormap as cm
from matplotlib import colors as colors
import math
import hashlib

def crop_raster(raster_path, bbox, output_raster_path):
    """ Creates a new raster cropped to the bbox
    """
    data_path = raster_path

    minx, miny, maxx, maxy = bbox
    boundary_box = box(minx, miny, maxx, maxy)

    with rasterio.open(data_path) as src:
        out_image, out_transform = rasterio.mask.mask(src, [boundary_box], crop=True)
        out_meta = src.meta

    out_meta.update({"driver": "GTiff",
                 "height": out_image.shape[1],
                 "width": out_image.shape[2],
                 "transform": out_transform})

    with rasterio.open(output_raster_path, "w", **out_meta) as dest:
        dest.write(out_image)

def add_raster_to_folium(raster_path, folium_map):
    # Open the raster file
    with rasterio.open(raster_path) as src:
        # Read the data
        img = src.read()
        # Get the bounds of the raster
        boundary = src.bounds
    
    img[img<0.0] = np.nan

    clat = (boundary.bottom + boundary.top)/2
    clon = (boundary.left + boundary.right)/2

    vmin = np.floor(np.nanmin(img))
    vmax = np.ceil(np.nanmax(img))

    colormap = cm.linear.RdBu_11.scale(vmin, vmax)

    def mapvalue2color(value, cmap): 
        """
        Map a pixel value of image to a color in the rgba format. 
        As a special case, nans will be mapped totally transparent.
        
        Inputs
            -- value - pixel value of image, could be np.nan
            -- cmap - a linear colormap from branca.colormap.linear
        Output
            -- a color value in the rgba format (r, g, b, a)    
        """
        if np.isnan(value) or value == 0:
            return (1, 0, 0, 0)
        else:
            return colors.to_rgba(cmap(value), 0.7)  
        
    folium.raster_layers.ImageOverlay(
        image=img[0],
        name='Population map',
        opacity=1,
        bounds= [[boundary.bottom, boundary.left], [boundary.top, boundary.right]],
        colormap= lambda value: mapvalue2color(value, colormap)
    ).add_to(folium_map)


    #
    colormap.caption = 'Population (number/ha)'
    folium_map.add_child(colormap)

    return folium_map

# Function to get the bounding box of a graph
def get_graph_bbox(G):
    nodes = G.nodes(data=True)
    
    y_values = [data['y'] for node, data in nodes]
    x_values = [data['x'] for node, data in nodes]

    north, south = max(y_values), min(y_values)
    east, west = max(x_values), min(x_values)

    return north, south, east, west 


# Function to estimate flows from origin to destinations using a 
# production-constrained gravity model
# NB: The origin attractivity is not used as it drops when normalizing
def prod_constrained_gravity_power(origin_n_trips, dest_attractivity_list, cost_list, gamma = 1):

    flows = np.zeros(len(dest_attractivity_list))
    norm_constant = .0

    # Calculating raw flows and normalisation constant
    for j in range(len(flows)):
        if cost_list[j] == 0:
            print(f"ALERT \t Cost function is NULL some fluxes, setting the corresponding flux to zero", end='\r')
            flows[j] = 0
            attractivity_over_cost = 0
        else:
            attractivity_over_cost = dest_attractivity_list[j] / (cost_list[j]**gamma)
            flows[j] = origin_n_trips * attractivity_over_cost

        norm_constant += attractivity_over_cost

    # Normalisation
    flows = flows / norm_constant

    return flows

# Function to estimate flows from origin to destinations using a 
# production-constrained gravity model with expontential cost
# NB: The origin attractivity is not used as it drops when normalizing
def prod_constrained_gravity_exp(origin_n_trips, dest_attractivity_list, cost_list, beta = 1):

    flows = np.zeros(len(dest_attractivity_list))
    norm_constant = .0

    # Calculating raw flows and normalisation constant
    for j in range(len(flows)):
        if cost_list[j] == 0:
            print(f"ALERT \t Cost function is NULL for some fluxes, setting the corresponding flux to zero", end='\r')
            flows[j] = 0
            attractivity_over_cost = 0
        else:
            attractivity_over_cost = dest_attractivity_list[j] / math.exp(beta * cost_list[j])
            flows[j] = origin_n_trips * attractivity_over_cost

        norm_constant += attractivity_over_cost

    # Normalisation
    flows = flows / norm_constant

    return flows

# Function to estimate flows from origin to destinations using a 
# production-constrained radiation model 
def prod_constrained_radiation(origin_n_trips, origin_attractivity, dest_attractivity_list, cost_list):

    # Step 1: Initialize variables
    flows = np.zeros(len(dest_attractivity_list))
    norm_constant = .0
    intervening_opportunity = .0

    # Step 2: Create list of tuples to store initial order of cost_list and order them by distance
    # Create a list of tuples (value, original_index)
    indexed_cost_list = list(enumerate(cost_list))

    # Sort the list of tuples based on the values
    sorted_indexed_cost_list = sorted(indexed_cost_list, key=lambda x: x[1])

    # Prepare a list to store flows with their original indices
    calculated_indexed_flows = []

    # Step 3: Calculate raw flows and normalization constant
    i = 0
    for original_index, _ in sorted_indexed_cost_list:
        j = original_index  # Use the original index from the sorted list

        num = dest_attractivity_list[j] # Origin attractivty cancels out when normalizating. Thus, it has been removed, also avoid zero values when origin attractivty is null
        den = (origin_attractivity + intervening_opportunity) * (origin_attractivity + dest_attractivity_list[j] + intervening_opportunity)

        if den == 0:
            print(f"ALERT \t Denominator of radiation model is NULL. Flow is set to zero.", end='\r')
            attractivity_over_cost = 0
            calculated_flow = 0
        else:
            attractivity_over_cost = num / den
            calculated_flow = origin_n_trips * attractivity_over_cost

        flows[j] = calculated_flow

        # Add intervening opportunity only if the next destination is farther away than the current one
        if i>=1 and (sorted_indexed_cost_list[i-1] < sorted_indexed_cost_list[i]):
            intervening_opportunity += dest_attractivity_list[j]

        norm_constant += attractivity_over_cost

        # Append the calculated flow with its original index to the list
        calculated_indexed_flows.append((original_index, flows[j]))

        i = i + 1

    # Step 4: Sort back to the original order using the original indices
    original_order_flows = sorted(calculated_indexed_flows, key=lambda x: x[0])

    # Extract the values from the tuples to get the final list
    final_flows = [value for index, value in original_order_flows]

    # Step 5: Normalize the flows
    final_flows = np.array(final_flows) / norm_constant

    return final_flows

def create_unique_id(variables):
    # Example list of variables
    # variables = [123, 'example', 45.67, 'another_example']

    # Convert the list of variables into a single string
    combined_string = '_'.join(map(str, variables))
    # Generate a unique hash of the combined string
    unique_id = hashlib.md5(combined_string.encode()).hexdigest()
    return unique_id