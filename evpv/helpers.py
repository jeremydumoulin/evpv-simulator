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


# Function to estimate flows using the radiation model
def radiation(T, dist_matrix, pop_matrix, pop_origin):

    def from_origin():
        # Sort destinations by distance from origin
        didxs = np.argsort(dist_matrix)
        pop_matrix_sorted =  pop_matrix[didxs]
        pop_in_radius = 0
        flows_proba = np.zeros(T.shape)
        for j in range(T.shape[0]):
            num = pop_origin*pop_matrix_sorted[j]
            denom = (pop_origin + pop_in_radius)*(pop_origin + pop_matrix_sorted[j] + pop_in_radius)
            flows_proba[j] = num/denom
            pop_in_radius += pop_matrix_sorted[j]
        # Unsort list

        return flows_proba[didxs.argsort()]

    # Builds the OD matrix T from the input data
    T_norm_p = np.zeros(T.shape)
    T_norm_p = from_origin()
    
    return T_norm_p