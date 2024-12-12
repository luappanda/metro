import os
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase
from matplotlib.font_manager import FontProperties
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
# Add a colorbar with explicit configuration
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm._A = []  # Dummy array for ScalarMappable
cbar = fig.colorbar(sm, ax=ax)

# Adjust the label and its distance from the colorbar
cbar.set_label('Total Weighted Feasibility', labelpad=20.0, fontsize=15.0)  # Increase labelpad as needed

# Add titles and labels
ax.set_title("Weighted Feasibility", fontsize=15.0)
ax.axis('off')

# Save the plot as a PNG file
plt.savefig(output_png, dpi=500, bbox_inches='tight')
plt.close()

print(f"Visualization saved to {output_png}")