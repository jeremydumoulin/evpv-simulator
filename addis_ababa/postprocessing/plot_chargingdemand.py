import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely import wkt
import matplotlib as mpl
import contextily as ctx
from matplotlib.colors import LinearSegmentedColormap

# Load the CSV files
regions_df = pd.read_csv("../output/ChargingDemand_Home_spatial_demand.csv")  # Replace with the actual path to the region CSV

# Preprocess geometries in the regions file
regions_df['geometry'] = regions_df['geometry'].apply(wkt.loads)
regions_gdf = gpd.GeoDataFrame(regions_df, geometry='geometry')
regions_gdf.set_crs("EPSG:4326", inplace=True)

# Calculate the aggregated charging demand (in MWh)
regions_gdf['Aggregated_Charging_Demand'] = (
    regions_gdf['Etot_home_kWh'] + 
    regions_gdf['Etot_work_kWh'] + 
    regions_gdf['Etot_poi_kWh']
) / 1000  # Converting from kWh to MWh

# Load boundary GeoJSON
boundary_gdf = gpd.read_file("../input/gadm41_ETH_1_AddisAbeba.json")  # Replace with the path to your boundary GeoJSON

# Normalize color scale
vmin = 0
vmax = 25
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

# Create a single plot
fig, ax = plt.subplots(figsize=(10, 8))

# Plot the light grey basemap using contextily
# First, reproject your regions GeoDataFrame to the same CRS as the basemap (EPSG:3857)
regions_gdf = regions_gdf.to_crs(epsg=3857)
boundary_gdf = boundary_gdf.to_crs(epsg=3857)

# Define the custom color scale (Green-blue or other options)
# colors = ["#E5FBFC", "#119EA2", "#021A1A"]  # Green-blue 
# colors = ["#FEF6F1", "#F08228", "#291402"]  # Orange
colors = ["#FCEEF7", "#C0238B", "#320924"]  # Purple

# Create the custom colormap
custom_cmap = LinearSegmentedColormap.from_list("custom_blue", colors, N=256)

# Plot the aggregated charging demand
regions_gdf.plot(
    column='Aggregated_Charging_Demand', cmap=custom_cmap, linewidth=0.5, ax=ax, edgecolor='0.8', legend=False, norm=norm
)

# Plot the boundary
boundary_gdf.plot(ax=ax, color='none', edgecolor='black', linewidth=2)

# Add the basemap (light grey CartoDB basemap)
ctx.add_basemap(ax, crs=regions_gdf.crs.to_string(), source=ctx.providers.CartoDB.Positron)

# Remove the title
ax.axis('off')

# Add the colorbar with a vertical label
cbar = fig.colorbar(ax.collections[0], ax=ax, orientation='vertical', fraction=0.02, pad=0.04)
cbar.set_label('Charging needs (MWh)', rotation=270, labelpad=20)

# Adjust layout to remove extra margins
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

# Show the plot
plt.tight_layout()
plt.show()
