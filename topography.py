import geopandas as gpd

# Load the grid data and topography data
grid_gdf = gpd.read_file('C:/Users/kavan_3rgiqdq/Documents/metro project/CountyGrid.gpkg')
topo_gdf = gpd.read_file('C:/Users/kavan_3rgiqdq/Documents/metro project/clipped topography.gpkg') 

# Ensure both datasets use the same CRS
if grid_gdf.crs != topo_gdf.crs:
    topo_gdf = topo_gdf.to_crs(grid_gdf.crs)

# Set a user-defined elevation threshold for feasibility
user_input_threshold = 520  # Replace with actual user input value

# Filter topography data based on the elevation threshold
filtered_gdf = topo_gdf[topo_gdf['elevation'] > user_input_threshold]

# Initialize the feasibility column in grid_gdf
grid_gdf['IS_FEASIBLE'] = 1  # Assume initially that all grids are feasible

# Perform spatial join to check for intersection with filtered topography
# (e.g., topography above the elevation threshold, which makes it unfeasible)
intersection_gdf = gpd.sjoin(grid_gdf, filtered_gdf, how="left", predicate="intersects")

# Identify grids that intersect with unfeasible elevation areas
unfeasible_indices = intersection_gdf[intersection_gdf['index_right'].notnull()].index.unique()
grid_gdf.loc[unfeasible_indices, 'IS_FEASIBLE'] = 0  # Mark as unfeasible

# Output the updated grid to a new file
output_fp = "C:/Users/kavan_3rgiqdq/Documents/metro project/feasibility_grid.gpkg"
grid_gdf.to_file(output_fp, driver="GPKG")

print("Topography feasibility grid processing complete.")
