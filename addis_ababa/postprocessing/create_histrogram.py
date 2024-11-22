import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV from a file
file_path = "output/MobilitySimulation_results_flows.csv"  # Replace with your actual file path
flows_df = pd.read_csv(file_path)

# Replicate the travel distance according to the flow
expanded_distances = np.repeat(flows_df['Travel Distance (km)'], flows_df['Flow'])

# Set the bin size (in kilometers)
bin_size = 0.5  # Change this value to set the desired bin size
max_distance = expanded_distances.max()
bins = np.arange(0, max_distance + bin_size, bin_size)

# Prepare histogram data
hist_values, bin_edges = np.histogram(expanded_distances, bins=bins)

# Create a DataFrame for the histogram data
hist_df = pd.DataFrame({
    'Bin Start (km)': bin_edges[:-1],
    'Bin End (km)': bin_edges[1:],
    'Flow-Weighted Count': hist_values
})

# Export the histogram data to a CSV file
hist_df.to_csv("flow_weighted_histogram.csv", index=False)

# Plot the histogram
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(
    bin_edges[:-1],
    hist_values,
    width=bin_size,
    align='edge',
    color='skyblue',
    edgecolor='black'
)

# Add labels and title
ax.set_xlabel('Travel Distance (km)', fontsize=12)
ax.set_ylabel('Total Flow in Distance Range (km)', fontsize=12)
ax.set_title('Histogram of Flow-Weighted Travel Distance by Road', fontsize=14)

# Add grid for better readability
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Show plot
plt.tight_layout()
plt.show()
