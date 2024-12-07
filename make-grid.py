import os
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase
import numpy as np

# Define file paths
input_gpkg = "GISFiles/weighted grid.gpkg"  # Replace with the path to your GeoPackage
output_png = "GISFiles/grid_visualization.png"

# Load the GeoPackage
gdf = gpd.read_file(input_gpkg)

# Ensure the GeoPackage contains the necessary data
if 'TOTAL WEIGHTED FEASIBILITY' not in gdf.columns:
    raise ValueError("The GeoPackage does not contain the requested column.")

# Create the plot
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Define a colormap and normalization for weights
cmap = plt.cm.Reds  # Choose a colormap
norm = Normalize(vmin=gdf['TOTAL WEIGHTED FEASIBILITY'].min(), vmax=gdf['TOTAL WEIGHTED FEASIBILITY'].max())

# Plot the GeoDataFrame with color based on the 'TOTAL WEIGHTED FEASIBILITY' column
gdf.plot(column='TOTAL WEIGHTED FEASIBILITY', cmap=cmap, norm=norm, ax=ax)

# Add a colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm._A = []  # Dummy array for ScalarMappable
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label('Total Weighted Feasbility')

# Add titles and labels
ax.set_title("Weighted Feasibility")
ax.axis('off')

# Save the plot as a PNG file
plt.savefig(output_png, dpi=500, bbox_inches='tight')
plt.close()

print(f"Visualization saved to {output_png}")