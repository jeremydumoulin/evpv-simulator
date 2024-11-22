import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely import wkt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl

# Load CSV and convert geometry column to Shapely polygons
df = pd.read_csv("output/region.csv")
df['geometry'] = df['geometry'].apply(wkt.loads)

# Create a GeoDataFrame and set CRS to WGS84
gdf = gpd.GeoDataFrame(df, geometry='geometry')
gdf.set_crs("EPSG:4326", inplace=True)

# Load the boundary from the GeoJSON file
boundary_gdf = gpd.read_file("input/gadm41_ETH_1_AddisAbeba.json")

# Set up the figure and axes for 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Adjust layout to create spacing
fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, wspace=0.2)

# Plot n_people with Reds color scheme and add the boundary
divider0 = make_axes_locatable(axes[0])
cax0 = divider0.append_axes("bottom", size="5%", pad=0.3)
gdf.plot(column='n_people', cmap='Reds', linewidth=0.5, ax=axes[0], edgecolor='0.8')
boundary_gdf.plot(ax=axes[0], color='none', edgecolor='black', linewidth=2)

# Create colorbar with specified range and minor ticks for population
norm_people = mpl.colors.Normalize(vmin=0, vmax=250000)
cb0 = plt.colorbar(mpl.cm.ScalarMappable(norm=norm_people, cmap='Reds'), cax=cax0, orientation="horizontal")
cb0.set_ticks([0, 50000, 100000, 150000, 200000, 250000])  # Major ticks
cb0.minorticks_on()
cb0.ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))  # Add minor ticks

# Remove title, labels, and axis edges
axes[0].axis('off')
# Add caption
axes[0].text(0.02, 0.98, '(a)', transform=axes[0].transAxes, fontsize=12, verticalalignment='top', weight='bold')

# Plot n_workplaces with Greens color scheme and add the boundary
divider1 = make_axes_locatable(axes[1])
cax1 = divider1.append_axes("bottom", size="5%", pad=0.3)
gdf.plot(column='n_workplaces', cmap='Greens', linewidth=0.5, ax=axes[1], edgecolor='0.8')
boundary_gdf.plot(ax=axes[1], color='none', edgecolor='black', linewidth=2)

# Create colorbar with specified range and minor ticks for workplaces
norm_workplaces = mpl.colors.Normalize(vmin=0, vmax=110)
cb1 = plt.colorbar(mpl.cm.ScalarMappable(norm=norm_workplaces, cmap='Greens'), cax=cax1, orientation="horizontal")
cb1.set_ticks([0, 20, 40, 60, 80, 100, 110])  # Major ticks
cb1.minorticks_on()
cb1.ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))  # Add minor ticks

# Remove title, labels, and axis edges
axes[1].axis('off')
# Add caption
axes[1].text(0.02, 0.98, '(b)', transform=axes[1].transAxes, fontsize=12, verticalalignment='top', weight='bold')

# Plot n_pois with Blues color scheme and add the boundary
divider2 = make_axes_locatable(axes[2])
cax2 = divider2.append_axes("bottom", size="5%", pad=0.3)
gdf.plot(column='n_pois', cmap='Blues', linewidth=0.5, ax=axes[2], edgecolor='0.8')
boundary_gdf.plot(ax=axes[2], color='none', edgecolor='black', linewidth=2)

# Create colorbar with specified range and minor ticks for pois
norm_pois = mpl.colors.Normalize(vmin=0, vmax=250)
cb2 = plt.colorbar(mpl.cm.ScalarMappable(norm=norm_pois, cmap='Blues'), cax=cax2, orientation="horizontal")
cb2.set_ticks([0, 50, 100, 150, 200, 250])  # Major ticks
cb2.minorticks_on()
cb2.ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))  # Add minor ticks

# Remove title, labels, and axis edges
axes[2].axis('off')
# Add caption
axes[2].text(0.02, 0.98, '(c)', transform=axes[2].transAxes, fontsize=12, verticalalignment='top', weight='bold')

# Show plot
plt.show()
