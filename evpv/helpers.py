# coding: utf-8

"""
Helper functions for the other classes of the evpv model

This module notably includes functions to facilitate operations on georeferenced objects and
folium maps. It also holds the formulation of the various spatial interaction models (gravity, 
raditation, ...) involved in mobility demand modelling.
"""

import numpy as np
import rasterio
from rasterio.mask import mask
from shapely.geometry import box
import folium
import branca.colormap as cm
import math
import hashlib

def crop_raster(raster_path: str, bbox: tuple[float, float, float, float], output_raster_path: str) -> None:
    """Creates a new raster cropped to the specified bounding box.

    Args:
        raster_path (str): The path to the input raster file.
        bbox (tuple): A tuple of four floats representing the bounding box (minx, miny, maxx, maxy).
        output_raster_path (str): The path where the output cropped raster will be saved.

    Returns:
        None
    """
    data_path = raster_path

    minx, miny, maxx, maxy = bbox
    boundary_box = box(minx, miny, maxx, maxy)

    with rasterio.open(data_path) as src:
        out_image, out_transform = rasterio.mask.mask(src, [boundary_box], crop=True)
        out_meta = src.meta

    out_meta.update({
        "driver": "GTiff",
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform
    })

    with rasterio.open(output_raster_path, "w", **out_meta) as dest:
        dest.write(out_image)


def add_raster_to_folium(raster_path: str, folium_map: folium.Map) -> folium.Map:
    """Adds a raster overlay to a Folium map.

    Args:
        raster_path (str): The path to the raster file to be added.
        folium_map (folium.Map): The Folium map instance to which the raster will be added.

    Returns:
        folium.Map: The updated Folium map with the raster overlay.
    """
    # Open the raster file
    with rasterio.open(raster_path) as src:
        # Read the data
        img = src.read()
        # Get the bounds of the raster
        boundary = src.bounds

    img[img < 0.0] = np.nan

    clat = (boundary.bottom + boundary.top) / 2
    clon = (boundary.left + boundary.right) / 2

    vmin = np.floor(np.nanmin(img))
    vmax = np.ceil(np.nanmax(img))

    colormap = cm.linear.RdBu_11.scale(vmin, vmax)

    def mapvalue2color(value: float, cmap: cm.LinearColormap) -> tuple[float, float, float, float]:
        """Maps a pixel value of the image to a color in RGBA format.
        """
        if np.isnan(value) or value == 0:
            return (1, 0, 0, 0)  # Transparent for NaN or zero values
        else:
            return colors.to_rgba(cmap(value), 0.7)  

    folium.raster_layers.ImageOverlay(
        image=img[0],
        name='Population map',
        opacity=1,
        bounds=[[boundary.bottom, boundary.left], [boundary.top, boundary.right]],
        colormap=lambda value: mapvalue2color(value, colormap)
    ).add_to(folium_map)

    # Set colormap caption
    colormap.caption = 'Population (number/ha)'
    folium_map.add_child(colormap)

    return folium_map


def get_graph_bbox(G) -> tuple[float, float, float, float]:
    """Gets the bounding box of a graph.

    Args:
        G: The graph from which to calculate the bounding box.

    Returns:
        tuple: A tuple of four floats representing (north, south, east, west) boundaries of the graph.
    """
    nodes = G.nodes(data=True)
    
    y_values = [data['y'] for node, data in nodes]
    x_values = [data['x'] for node, data in nodes]

    north, south = max(y_values), min(y_values)
    east, west = max(x_values), min(x_values)

    return north, south, east, west


def prod_constrained_gravity_power(origin_n_trips: float, 
                                dest_attractivity_list: list[float], 
                                cost_list: list[float], 
                                gamma: float) -> np.ndarray:
    r"""Estimates flows from origin to destinations using a production-constrained gravity model with a power law cost.

    This is a special case of a spatial interaction model, where the total number of outflows $T_{i,out}$ 
    from origin $i$ is known (production-constrained), thus imposing the value of the constant $C_i$, and
    the cost function $f(d_{ij})$ follows a power law

    .. math::
        T_{ij} = T_{i,out} C_i A_j f(d_{ij})
    .. math::
        C_i = 1/\sum_{k \neq i} A_j f(d_{ij})
    .. math::
        f(d_{ij})=d_{ij}^{-\gamma}

    with $T_{ij}$ the number of trips from origin $i$ to destination $j$, and $A_j$ the attractivity 
    of destination $j$. Note note that $T_{ii}$ can not be calculated with this model as $d_{ii}$ is 
    generally not well defined.

    For an introduction to spatial interaction models, we recommend the following book chapter:  
    Griffith, D.A., Fischer, M.M. (2016). Constrained Variants of the Gravity Model and Spatial 
    Dependence: Model Specification and Estimation Issues. https://doi.org/10.1007/978-3-319-30196-9_3

    Args:
        origin_n_trips (float): The total number of trips originating from the origin.
        dest_attractivity_list (list[float]): A list of attractivity values for each destination.
        cost_list (list[float]): A list of cost values corresponding to each destination.
        gamma (float, optional): The exponent applied to the cost in the calculation. Defaults to 1.

    Returns:
        np.ndarray: An array of estimated flows to each destination.
    """
    flows = np.zeros(len(dest_attractivity_list))
    norm_constant = .0

    # Calculating raw flows and normalization constant
    for j in range(len(flows)):
        if cost_list[j] == 0:
            print(f"ALERT \t Cost function is NULL for some fluxes, setting the corresponding flux to zero", end='\r')
            flows[j] = 0
            attractivity_over_cost = 0
        else:
            attractivity_over_cost = dest_attractivity_list[j] / (cost_list[j] ** gamma)
            flows[j] = origin_n_trips * attractivity_over_cost

        norm_constant += attractivity_over_cost

    # Normalization
    flows = flows / norm_constant

    return flows


def prod_constrained_gravity_exp(origin_n_trips: float, 
                                  dest_attractivity_list: list[float], 
                                  cost_list: list[float], 
                                  beta: float) -> np.ndarray:
    r"""Estimates flows from origin to destinations using a production-constrained gravity model with exponential cost.

    This is a special case of a spatial interaction model, where the cost function$f(d_{ij})$ follows an exponential 
    law.

    .. math::
        f(d_{ij})=e^{-d_{ij} \beta}

    See prod_constrained_gravity_power() for more details on spatial interaction models for trip distribution.

    Args:
        origin_n_trips (float): The total number of trips originating from the origin.
        dest_attractivity_list (list[float]): A list of attractivity values for each destination.
        cost_list (list[float]): A list of cost values corresponding to each destination.
        beta (float, optional): The exponent applied to the cost in the calculation. 

    Returns:
        np.ndarray: An array of estimated flows to each destination.
    """
    flows = np.zeros(len(dest_attractivity_list))
    norm_constant = .0

    # Calculating raw flows and normalization constant
    for j in range(len(flows)):
        if cost_list[j] == 0:
            print(f"ALERT \t Cost function is NULL for some fluxes, setting the corresponding flux to zero", end='\r')
            flows[j] = 0
            attractivity_over_cost = 0
        else:
            attractivity_over_cost = dest_attractivity_list[j] / math.exp(beta * cost_list[j])
            flows[j] = origin_n_trips * attractivity_over_cost

        norm_constant += attractivity_over_cost

    # Normalization
    flows = flows / norm_constant

    return flows

def prod_constrained_radius(origin_n_trips: float, 
                            dest_attractivity_list: list[float], 
                            cost_list: list[float], 
                            radius: float = 10) -> np.ndarray:
    """Estimates flows from origin to destinations using a production-constrained model based on distance radius.

    This is a simplified example of a spatial interaction model. In this case, the product of attractiveness 
    and cost is set to 1 if the destination falls within a specified radius, and 0 otherwise. This function 
    is purely hypothetical and has no practical application justification. It can serve as a test function 
    since it directs all flows to destinations within the given radius.

    See prod_constrained_gravity_power() for more details on spatial interaction models for trip distribution.

    Args:
        origin_n_trips (float): The total number of trips originating from the origin.
        dest_attractivity_list (list[float]): A list of attractivity values for each destination.
        cost_list (list[float]): A list of cost values corresponding to each destination.
        radius (float, optional): The maximum distance for attractivity. 

    Returns:
        np.ndarray: An array of estimated flows to each destination.
    """
    flows = np.zeros(len(dest_attractivity_list))
    norm_constant = .0

    # Calculating raw flows and normalization constant
    for j in range(len(flows)):
        if cost_list[j] == 0:
            print(f"ALERT \t Cost function is NULL for some fluxes, setting the corresponding flux to zero", end='\r')
            flows[j] = 0
            attractivity_over_cost = 0
        else:
            if cost_list[j] <= radius:
                attractivity_over_cost = 1
            else:
                attractivity_over_cost = 0
            flows[j] = origin_n_trips * attractivity_over_cost

        norm_constant += attractivity_over_cost

    # Normalization
    flows = flows / norm_constant

    return flows


def prod_constrained_radiation(origin_n_trips: float, 
                               origin_attractivity: float, 
                               dest_attractivity_list: list[float], 
                               cost_list: list[float]) -> np.ndarray:
    r"""Estimates flows from origin to destinations using a production-constrained radiation model.

    This is a special case of an intervening opportunity spatial interaction model. In this case, the algorithm is more
    complex than gravity laws, as the cost function $f(d_{ij})$ includes the intervening opportunities located at a distance 
    smaller than $d_{ij}$ 

    .. math::
        f(d_{ij}) = ((A_i+s_{ij}).(A_i+A_j+s_{ij}))^{-1}
    .. math::
        s_{ij} = \sum_{k \neq i,j} A_k, ~ if ~ d_{ik}<d_{ij} 

    with $s_{ij}$ being the intervening opportunities. Refer to prod_constrained_gravity_power() for other symbols.

    To our knowledge, this model has been initially described in the following publication:
    Simini, F., González, M., Maritan, A. et al. A universal model for mobility and migration patterns. Nature 
    484, 96–100 (2012). https://doi.org/10.1038/nature10856.

    While this model has the advantage of being parameter-free, it seems to not perform well at smaller scales.

    Args:
        origin_n_trips (float): The total number of trips originating from the origin.
        origin_attractivity (float): The attractivity of the origin.
        dest_attractivity_list (list[float]): A list of attractivity values for each destination.
        cost_list (list[float]): A list of cost values corresponding to each destination.

    Returns:
        np.ndarray: An array of estimated flows to each destination.
    """
    # Step 1: Initialize variables
    flows = np.zeros(len(dest_attractivity_list))
    norm_constant = .0
    intervening_opportunity = .0

    # Step 2: Create list of tuples to store initial order of cost_list and order them by distance
    indexed_cost_list = list(enumerate(cost_list))

    # Sort the list of tuples based on the values
    sorted_indexed_cost_list = sorted(indexed_cost_list, key=lambda x: x[1])

    # Prepare a list to store flows with their original indices
    calculated_indexed_flows = []

    # Step 3: Calculate raw flows and normalization constant
    i = 0
    for original_index, _ in sorted_indexed_cost_list:
        j = original_index  # Use the original index from the sorted list

        num = dest_attractivity_list[j]  # Origin attractivity cancels out when normalizing.
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
        if i >= 1 and (sorted_indexed_cost_list[i - 1] < sorted_indexed_cost_list[i]):
            intervening_opportunity += dest_attractivity_list[j]

        norm_constant += attractivity_over_cost

        # Append the calculated flow with its original index to the list
        calculated_indexed_flows.append((original_index, flows[j]))

        i += 1

    # Step 4: Sort back to the original order using the original indices
    original_order_flows = sorted(calculated_indexed_flows, key=lambda x: x[0])

    # Extract the values from the tuples to get the final list
    final_flows = [value for index, value in original_order_flows]

    # Step 5: Normalize the flows
    final_flows = np.array(final_flows) / norm_constant

    return final_flows


def create_unique_id(variables: list) -> str:
    """Creates a unique identifier based on the input variables.

    Args:
        variables (list): A list of variables to generate a unique ID from.

    Returns:
        str: A unique hash string representing the combined input variables.
    """
    # Convert the list of variables into a single string
    combined_string = '_'.join(map(str, variables))

    # Generate a unique hash of the combined string
    unique_id = hashlib.md5(combined_string.encode()).hexdigest()

    return unique_id

def calculate_days_between_charges_single_vehicle(
    daily_charging_demand: float, 
    battery_capacity: float, 
    soc_threshold_mean: float = 0.6, 
    soc_threshold_std_dev: float = 0.2
) -> float:
    """ 
    Calculates the average number of days between charges for a vehicle based on its daily energy demand and battery capacity.
    
    The number of days is calculated using a random state of charge (SoC) threshold sampled from a normal distribution (as described in Pareschi et al., Applied Energy, 2020).
    The average days between charges is determined by the SoC threshold and the daily rate of SoC decrease.
    
    Args:
        daily_charging_demand (float): The daily energy demand of the vehicle in kWh.
        battery_capacity (float): The total battery capacity of the vehicle in kWh.
        soc_threshold_mean (float, optional): The mean value of the SoC threshold. Defaults to 0.6 (as described in Pareschi et al., Applied Energy, 2020).
        soc_threshold_std_dev (float, optional): The standard deviation of the SoC threshold. Defaults to 0.2 (as described in Pareschi et al., Applied Energy, 2020).
    
    Returns:
        float: The average number of days between full charges. The value is capped at a minimum of 1 day between two charges.
    """
    # 0. Useful battery capacity (Pareschi et al., Applied Energy, 2020)
    battery_capacity = battery_capacity * 0.8
    
    # 1. Randomly assign a state of charge (SoC) threshold from a normal distribution
    soc_threshold = np.random.normal(loc=soc_threshold_mean, scale=soc_threshold_std_dev)
    
    # Ensure SoC threshold is between 0 and 1
    soc_threshold = np.clip(soc_threshold, 0, 1)

    # 2. Calculate the daily decrease in SoC based on the daily energy demand and battery capacity
    daily_soc_decrease = daily_charging_demand / battery_capacity

    # 3. Calculate the average number of days between two charges
    if daily_soc_decrease > 0:  # Prevent division by zero
        days = (1 - soc_threshold) / daily_soc_decrease
    else:
        days = 1  # If daily demand is zero, set the number of days to 1

    # Ensure that the days between charges is at least 1
    return max(days, 1)
