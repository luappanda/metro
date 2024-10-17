import geopandas as gpd

# Step 1: Load AADT polyline features and grids (replace with your actual file paths)
traffic_gdf = gpd.read_file('C:/Users/kavan_3rgiqdq/Documents/metro project/clipped traffic.gpkg')  # Annual Average Daily Traffic data as polylines
grid_gdf = gpd.read_file('C:/Users/kavan_3rgiqdq/Documents/metro project/CountyGrid.gpkg')  # Grid data

# Ensure both datasets have the same CRS
if grid_gdf.crs != traffic_gdf.crs:
    traffic_gdf = traffic_gdf.to_crs(grid_gdf.crs)

# Step 2: Filter AADT features based on user input threshold
user_input_threshold = 10000  # Replace with actual user input value
filtered_gdf = traffic_gdf[traffic_gdf['year_2022'] > user_input_threshold]

# Initialize field to indicate which grid blocks contain stations
grid_gdf['IS_FEASIBLE'] = 0

# Perform spatial join to find intersections
intersection_gdf = gpd.sjoin(grid_gdf, filtered_gdf, how="left", predicate="intersects")

# Get grid cells that do not intersect traffic
feasible_gdf = intersection_gdf[intersection_gdf['index_right'].isnull()]

# Drop the 'index_right' column to avoid conflicts during the next join
feasible_gdf = feasible_gdf.drop(columns=['index_right'], errors='ignore')

# Create buffers around "feasible" grid cells
buffer_distance = 1000  # Adjust this as necessary based on the coordinate reference system
feasible_gdf.loc[:, 'geometry'] = feasible_gdf.geometry.buffer(buffer_distance)

# Perform spatial join with the buffered geometries
buffered_intersection_gdf = gpd.sjoin(filtered_gdf, feasible_gdf, how="left", predicate="intersects")

# Get unique indices of grid blocks that are within the buffer
buffered_station_indices = buffered_intersection_gdf[buffered_intersection_gdf['index_right'].notnull()].index.unique()

# Mark feasible grid blocks in the original grid_gdf
grid_gdf.loc[buffered_station_indices, 'IS_FEASIBLE'] = 1

# Create an output path for the data
output_fp = "C:/Users/kavan_3rgiqdq/Documents/metro project/traffic grid.gpkg"

# Write the result to a file
grid_gdf.to_file(output_fp, driver="GPKG")