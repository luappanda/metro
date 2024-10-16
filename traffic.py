import geopandas as gpd

# Get a GeoDataFrame of the grid and traffic
grid_gdf = gpd.read_file("C:/Users/kavan_3rgiqdq/Documents/metro project/CountyGrid.gpkg")
traffic_gdf = gpd.read_file("C:/Users/kavan_3rgiqdq/Documents/metro project/clipped traffic.gpkg")

# Ensure both GeoDataFrames have the same CRS
if grid_gdf.crs != traffic_gdf.crs:
    traffic_gdf = traffic_gdf.to_crs(grid_gdf.crs)
    
# Filter traffic data based on user input for AADT
# Assume the AADT values are stored in a column named 'AADT'
user_input_aadt_threshold = 10000  # Replace this with user input as needed
filtered_traffic_gdf = traffic_gdf[traffic_gdf['year_2023'] > user_input_aadt_threshold]

# Initialize a field to indicate grid feasibility (1 = feasible, 0 = not feasible)
grid_gdf['FEASIBLE'] = 1  # Assume all grid blocks are feasible initially

# Define a buffer radius for the grid blocks (e.g., RA)
buffer_radius = 1800  # Replace this with the desired buffer radius based on your grid's CRS

# Create buffers around each grid block
grid_gdf['geometry'] = grid_gdf.geometry.buffer(buffer_radius)

# Perform a spatial join to find intersections between the buffered grid blocks and filtered traffic features
intersection_gdf = gpd.sjoin(grid_gdf, filtered_traffic_gdf, how="left", predicate="intersects")

# Identify grid blocks that intersect with busy roads (AADT above the threshold)
intersected_grid_indices = intersection_gdf[intersection_gdf['index_right'].notnull()].index.unique()

# Mark these grid blocks as not feasible (0)
grid_gdf.loc[intersected_grid_indices, 'FEASIBLE'] = 0

# Create an output path for the processed grid data
output_fp = "C:/Users/kavan_3rgiqdq/Documents/metro project/grid_traffic_feasibility.gpkg"

# Write the updated grid GeoDataFrame to file
grid_gdf.to_file(output_fp, driver="GPKG")
