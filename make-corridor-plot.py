import geopandas as gpd
import matplotlib.pyplot as plt
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Define the file paths
gpkg_corridor1_path = r'GISFiles/output/corridor.gpkg'
gpkg_corridor2_path = r'GISFiles/output/corridor2.gpkg'
gpkg_corridor3_path = r'GISFiles/output/corridor3.gpkg'
gpkg_population_path = r'GISFiles/block population.gpkg'

# Load the GeoPackages
gdf_corridor1 = gpd.read_file(gpkg_corridor1_path)
gdf_corridor2 = gpd.read_file(gpkg_corridor2_path)
gdf_corridor3 = gpd.read_file(gpkg_corridor3_path)
gdf_population = gpd.read_file(gpkg_population_path)

wgs84_crs = "EPSG:4326"
gdf_corridor1 = gdf_corridor1.to_crs(wgs84_crs)
gdf_corridor2 = gdf_corridor2.to_crs(wgs84_crs)
gdf_corridor3 = gdf_corridor3.to_crs(wgs84_crs)
gdf_population = gdf_population.to_crs(wgs84_crs)

# Plotting
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Plot the population data first (background)
# We add transparency by setting the color to 'none' and a light border
gdf_population.plot(ax=ax, color='none', edgecolor='lightgray', linewidth=0.5)

# Plot the traffic data on top
gdf_corridor1.plot(ax=ax, color='green', alpha=0.5)
gdf_corridor2.plot(ax=ax, color='blue', alpha=0.5)
gdf_corridor3.plot(ax=ax, color='red', alpha=0.5)

# Create custom legend handles
legend_handles = [
    Line2D([0], [0], color='green', lw=2, label='Corridor 1'),
    Line2D([0], [0], color='blue', lw=2, label='Corridor 2'),
    Line2D([0], [0], color='red', lw=2, label='Corridor 3')
]

# Add title, labels, and legend
ax.set_title('Corridors Generated for All Metro Lines', fontsize=16)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.legend(handles=legend_handles, loc='lower left', title='Corridors')

# Save the plot
output_path = r'GISFiles/output/corridors.png'
plt.savefig(output_path, dpi=500, bbox_inches='tight')
# plt.show()