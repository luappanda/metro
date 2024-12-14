import geopandas as gpd
import matplotlib.pyplot as plt
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Define the file paths
gpkg_stations1 = r'GISFiles/output/best stations4.gpkg'
gpkg_stations2 = r'GISFiles/output/best stations2.gpkg'
gpkg_stations3 = r'GISFiles/output/best stations3.gpkg'
gpkg_line1 = r'GISFiles/output/connections1.gpkg'
gpkg_line2 = r'GISFiles/output/connections2.gpkg'
gpkg_line3 = r'GISFiles/output/connections3.gpkg'
gpkg_population_path = r'GISFiles/block population.gpkg'

# Load the GeoPackages
gdf_stations1 = gpd.read_file(gpkg_stations1)
gdf_stations2 = gpd.read_file(gpkg_stations2)
gdf_stations3 = gpd.read_file(gpkg_stations3)
gdf_line1 = gpd.read_file(gpkg_line1)
gdf_line2 = gpd.read_file(gpkg_line2)
gdf_line3 = gpd.read_file(gpkg_line3)
gdf_population = gpd.read_file(gpkg_population_path)

# print(gdf_population.crs)

# Reproject all data to WGS84 (latitude/longitude)
wgs84_crs = "EPSG:4326"
gdf_stations1 = gdf_stations1.to_crs(wgs84_crs)
gdf_stations2 = gdf_stations2.to_crs(wgs84_crs)
gdf_stations3 = gdf_stations3.to_crs(wgs84_crs)
gdf_line1 = gdf_line1.to_crs(wgs84_crs)
gdf_line2 = gdf_line2.to_crs(wgs84_crs)
gdf_line3 = gdf_line3.to_crs(wgs84_crs)
gdf_population = gdf_population.to_crs(wgs84_crs)


# Plotting
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Plot the population data first (background)
# We add transparency by setting the color to 'none' and a light border
gdf_population.plot(ax=ax, color='none', edgecolor='lightgray', linewidth=0.5)

# Plot the traffic data on top
scaled_stations1 = gdf_stations1.copy()
scaled_stations1['geometry'] = scaled_stations1['geometry'].scale(xfact=5, yfact=5, origin='center')
scaled_stations2 = gdf_stations2.copy()
scaled_stations2['geometry'] = scaled_stations2['geometry'].scale(xfact=5, yfact=5, origin='center')
scaled_stations3 = gdf_stations3.copy()
scaled_stations3['geometry'] = scaled_stations3['geometry'].scale(xfact=5, yfact=5, origin='center')
scaled_stations1.plot(ax=ax, color='green', edgecolor='black', linewidth=1, zorder=5)
scaled_stations2.plot(ax=ax, color='blue', edgecolor='black', linewidth=1, zorder=5)
scaled_stations3.plot(ax=ax, color='red', edgecolor='black', linewidth=1, zorder=5)
gdf_line1.plot(ax=ax, color='green', linewidth=5)
gdf_line2.plot(ax=ax, color='blue', linewidth=5)
gdf_line3.plot(ax=ax, color='red', linewidth=5)

# Create custom legend handles
legend_handles = [
    Line2D([0], [0], color='green', lw=2, label='Line 1'),
    Line2D([0], [0], color='blue', lw=2, label='Line 2'),
    Line2D([0], [0], color='red', lw=2, label='Line 3')
]

# Add title, labels, and legend
ax.set_title('All Metro Lines', fontsize=16)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.legend(handles=legend_handles, loc='lower left', title='Lines')

# Save the plot
output_path = r'GISFiles/output/lines.png'
plt.savefig(output_path, dpi=500, bbox_inches='tight')
# plt.show()