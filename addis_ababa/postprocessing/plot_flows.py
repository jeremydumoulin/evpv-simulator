import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely import wkt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl

# Load the first CSV
flows_df = pd.read_csv("output/MobilitySimulation_results_flows.csv")  # Replace with the actual path to the first CSV
regions_df = pd.read_csv("output/region.csv")  # Replace with the actual path to the second CSV

# Preprocess geometries in the regions file
regions_df['geometry'] = regions_df['geometry'].apply(wkt.loads)
regions_gdf = gpd.GeoDataFrame(regions_df, geometry='geometry')
regions_gdf.set_crs("EPSG:4326", inplace=True)

# Calculate total outgoing flow for each origin
outgoing_flows = flows_df.groupby('Origin')['Flow'].sum().reset_index()
outgoing_flows.rename(columns={'Flow': 'Total_Outgoing_Flow'}, inplace=True)

# Merge outgoing flows with geometries
regions_with_flows = regions_gdf.merge(outgoing_flows, left_on='id', right_on='Origin', how='left')

# User-specified origins to highlight
selected_origins = ["8_10", "1_7"]  # Replace with user-chosen Origin IDs

# Plot total outgoing flows for all origins
fig, axes = plt.subplots(1, 3, figsize=(20, 8))

# Normalize color scale for all plots
vmin = regions_with_flows['Total_Outgoing_Flow'].min()
vmax = regions_with_flows['Total_Outgoing_Flow'].max()
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

# Add a boundary (optional: replace with your boundary file if needed)
boundary_gdf = gpd.read_file("input/gadm41_ETH_1_AddisAbeba.json")  # Replace with the path to your boundary GeoJSON

# Plot the total outgoing flows for all regions
regions_with_flows.plot(
    column='Total_Outgoing_Flow', cmap='Oranges', linewidth=0.5, ax=axes[0], edgecolor='0.8', legend=False
)
boundary_gdf.plot(ax=axes[0], color='none', edgecolor='black', linewidth=2)
axes[0].set_title("Total Outgoing Flow (All Origins)", fontsize=14)
axes[0].axis('off')

# Add custom colorbar for the first plot
sm = mpl.cm.ScalarMappable(cmap='Oranges', norm=norm)
sm.set_array([])
divider0 = make_axes_locatable(axes[0])
cax0 = divider0.append_axes("bottom", size="5%", pad=0.3)
cb0 = plt.colorbar(sm, cax=cax0, orientation='horizontal')
cb0.set_label("Total Outgoing Flow", fontsize=12)

# Function to plot specific origin's outgoing flows and highlight in red, with colorbar
def plot_specific_origin(ax, origin_id, color_map):
    # Filter flows from the specified origin
    flows_from_origin = flows_df[flows_df['Origin'] == origin_id]
    regions_with_flows_subset = regions_gdf.merge(
        flows_from_origin, left_on='id', right_on='Destination', how='left'
    )
    # Plot flows
    norm_specific = mpl.colors.Normalize(
        vmin=flows_from_origin['Flow'].min(),
        vmax=flows_from_origin['Flow'].max()
    )
    regions_with_flows_subset.plot(
        column='Flow', cmap=color_map, linewidth=0.5, ax=ax, edgecolor='0.8', legend=False
    )
    boundary_gdf.plot(ax=ax, color='none', edgecolor='black', linewidth=2)
    # Highlight the origin in red
    origin_geometry = regions_gdf[regions_gdf['id'] == origin_id]
    origin_geometry.plot(ax=ax, color='none', edgecolor='red', linewidth=2)
    # Customize plot
    ax.set_title(f"Outgoing Flows from Origin {origin_id}", fontsize=14)
    ax.axis('off')
    # Add colorbar
    sm_specific = mpl.cm.ScalarMappable(cmap=color_map, norm=norm_specific)
    sm_specific.set_array([])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.3)
    plt.colorbar(sm_specific, cax=cax, orientation='horizontal').set_label("Flow")

# Plot flows from the first user-specified origin
plot_specific_origin(axes[1], selected_origins[0], 'Blues')

# Plot flows from the second user-specified origin
plot_specific_origin(axes[2], selected_origins[1], 'Blues')

# Adjust layout
fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.95, wspace=0.3)

# Add captions
axes[0].text(0.02, 0.98, '(a)', transform=axes[0].transAxes, fontsize=12, verticalalignment='top', weight='bold')
axes[1].text(0.02, 0.98, '(b)', transform=axes[1].transAxes, fontsize=12, verticalalignment='top', weight='bold')
axes[2].text(0.02, 0.98, '(c)', transform=axes[2].transAxes, fontsize=12, verticalalignment='top', weight='bold')

# Show the plot
plt.show()
