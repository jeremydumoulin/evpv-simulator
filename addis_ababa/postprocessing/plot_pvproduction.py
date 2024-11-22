import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV data
file_path = "output/PVProduction.csv"  # Replace with the path to your CSV file
df = pd.read_csv(file_path, parse_dates=True, index_col=0)

# Ensure the index is in datetime format
df.index = pd.to_datetime(df.index)

# Extract day and hour
df["Day"] = df.index.dayofyear  # Day of the year (1–365/366)
df["Hour"] = df.index.hour      # Hour of the day

# Extract months
df["Month"] = df.index.strftime("%b")  # Abbreviated month names

# Pivot table for heatmap
pivot_table = df.pivot(index="Hour", columns="Day", values="PV Production (W/m2)")

# Create the heatmap
plt.figure(figsize=(12, 8))
plt.imshow(pivot_table, aspect='auto', cmap="YlGnBu", origin='lower')

# Add colorbar
cbar = plt.colorbar()
cbar.set_label("PV Production (W/m²)")

# Label the axes
plt.title("PV Production Heatmap")
plt.xlabel("Month")
plt.ylabel("Hour of the Day")

# Determine the x-axis tick positions and labels
first_day_of_month = df[df.index.is_month_start]  # Filter for the first day of each month
month_labels = list(first_day_of_month.index.strftime("%b"))  # Abbreviated months
month_positions = list(first_day_of_month["Day"].values - 1)  # X-axis positions (0-based indexing)

# Add a tick for the next January
month_labels.append("Jan")
month_positions.append(df["Day"].max())  # Position at the very end of the year

plt.xticks(ticks=month_positions, labels=month_labels, rotation=0)  # One tick per month + next January

# Set the y-axis limits to 5 and 22
plt.ylim(4, 22)

# Adjust the y-axis ticks to show only the selected range
plt.yticks(ticks=np.arange(5, 23, 2), labels=np.arange(5, 23, 2))  # Hours on y-axis (5 AM to 10 PM)

# Show the plot
plt.tight_layout()
plt.show()
# Show the plot
plt.tight_layout()
plt.show()