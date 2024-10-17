import geopandas as gpd

# Step 1: Load AADT polyline features and grids (replace with your actual file paths)
traffic_gdf = gpd.read_file('C:/Users/kavan_3rgiqdq/Documents/metro project/clipped traffic.gpkg')  # Annual Average Daily Traffic data as polylines
grid_gdf = gpd.read_file('C:/Users/kavan_3rgiqdq/Documents/metro project/CountyGrid.gpkg')  # Grid data
if grid_gdf.crs != traffic_gdf.crs:
    traffic_gdf = traffic_gdf.to_crs(grid_gdf.crs)

# Step 2: Filter AADT features based on user input threshold
user_input_threshold = 10000  # Replace with actual user input value
filtered_gdf = traffic_gdf[traffic_gdf['AADT'] > user_input_threshold]

# Step 3: Create buffer area around each grid cell
buffer_radius = 500  # Replace with actual buffer radius (in the same units as your grid)
grid_gdf['buffer_area'] = grid_gdf.geometry.buffer(buffer_radius)

# Step 4: Identify grids that are far away from AADT features
def check_intersection(grid_gdf, traffic_gdf):
    # Check if the grid buffer intersects with any AADT feature
    return grid_gdf.intersects(traffic_gdf.unary_union)

# Apply the check to each grid's buffer
grid_gdf['far_from_aadt'] = grid_gdf['buffer_area'].apply(lambda buf: not check_intersection(buf, filtered_gdf))

# Step 5: Set S_p^i = 0 for grids far from AADT features
grid_gdf['S_p_i'] = grid_gdf['far_from_aadt'].apply(lambda x: 0 if x else 1)

# Create a output path for the data
output_fp = "C:/Users/kavan_3rgiqdq/Documents/metro project/traffic grid.gpkg"

# Write the file
grid_gdf.to_file(output_fp, driver="GPKG")