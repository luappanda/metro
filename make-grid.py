import geopandas as gpd
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize

# Define the file paths
gpkg_traffic_path = r'GISFiles/clipped traffic.gpkg'
gpkg_population_path = r'GISFiles/block population.gpkg'

# Check if the files exist
if not os.path.exists(gpkg_traffic_path):
    raise FileNotFoundError(f"The traffic file {gpkg_traffic_path} does not exist.")
if not os.path.exists(gpkg_population_path):
    raise FileNotFoundError(f"The population file {gpkg_population_path} does not exist.")

# Load the GeoPackages
gdf_traffic = gpd.read_file(gpkg_traffic_path)
gdf_population = gpd.read_file(gpkg_population_path)

wgs84_crs = "EPSG:4326"
gdf_traffic = gdf_traffic.to_crs(wgs84_crs)
gdf_population = gdf_population.to_crs(wgs84_crs)

# Check if 'year_2023' column exists and remove NaN or Inf values
if 'year_2023' not in gdf_traffic.columns:
    raise KeyError("'year_2023' column not found in the traffic GeoDataFrame.")

# Remove NaN and Inf values from the traffic data
gdf_traffic = gdf_traffic[gdf_traffic['year_2023'].notna()]
gdf_traffic = gdf_traffic[~gdf_traffic['year_2023'].isin([np.inf, -np.inf])]

# Ensure valid geometries in the traffic data
gdf_traffic = gdf_traffic[gdf_traffic.is_valid]

# Check for empty GeoDataFrame (traffic)
if gdf_traffic.empty:
    raise ValueError("The filtered traffic GeoDataFrame is empty. Check the geometry validity or the 'year_2023' data.")

# Set up color map for traffic data
colors = ["green", "yellow", "orange", "red", "red"]
cmap = LinearSegmentedColormap.from_list("green_red", colors, N=256)

# Use Normalize instead of BoundaryNorm for continuous color mapping
norm = Normalize(vmin=gdf_traffic['year_2023'].min(), vmax=gdf_traffic['year_2023'].max())

# Plotting
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Plot the population data first (background)
# We add transparency by setting the color to 'none' and a light border
gdf_population.plot(ax=ax, color='none', edgecolor='lightgray', linewidth=0.5)

# Plot the traffic data on top
gdf_traffic.plot(ax=ax, column='year_2023', cmap=cmap, norm=norm, linewidth=2)

# Create the colorbar axis with a fixed position
cbar_ax = fig.add_axes([0.90, 0.15, 0.03, 0.7])  # Adjust position if necessary

# Manually set limits for the colorbar to avoid NaN or Inf issues
colorbar = ColorbarBase(cbar_ax, cmap=cmap, norm=norm, orientation='vertical')
colorbar.set_label('ADT Value')

# Add titles and labels
ax.set_title('Traffic Visualization by Linear ADT in Pulaski County', fontsize=16)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

# Save the plot
output_path = r'GISFiles/traffic_visualization.png'
plt.savefig(output_path, dpi=500, bbox_inches='tight')
plt.show()