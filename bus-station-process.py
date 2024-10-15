import geopandas as gpd

# Get a GeoDataFrame of the grid and stations
grid_gdf = gpd.read_file("C:/Users/Paul/Documents/metro project/county grid.gpkg")
stations_gdf = gpd.read_file("C:/Users/Paul/Documents/metro project/bus stations.gpkg")

# Ensure both GeoDataFrames have the same CRS
if grid_gdf.crs != stations_gdf.crs:
    stations_gdf = stations_gdf.to_crs(grid_gdf.crs)

# Initialize field to indicate which grid blocks contain stations
grid_gdf['CONTAINS_STATION'] = 0

# Perform spatial join to find intersections
intersection_gdf = gpd.sjoin(grid_gdf, stations_gdf, how="left", predicate="intersects")

# Get unique indices of grid blocks that contain stations
station_indices = intersection_gdf[intersection_gdf['index_right'].notnull()].index.unique()
grid_gdf.loc[station_indices, 'CONTAINS_STATION'] = 1

# Create buffers around station locations (e.g., radius of 2 grid blocks)
# You may need to adjust the distance based on the coordinate reference system
buffer_distance = 1800  # adjust this as necessary based on your grid's units
# 1800 for walking is good
# 4800 for biking
stations_gdf['geometry'] = stations_gdf.geometry.buffer(buffer_distance)

# Perform spatial join with the buffered geometries
buffered_intersection_gdf = gpd.sjoin(grid_gdf, stations_gdf, how="left", predicate="intersects")

# Get unique indices of grid blocks that are within the buffer
buffered_station_indices = buffered_intersection_gdf[buffered_intersection_gdf['index_right'].notnull()].index.unique()
grid_gdf.loc[buffered_station_indices, 'CONTAINS_STATION'] = 1  # or use a different field if needed

# Create an output path for the data
output_fp = "C:/Users/Paul/Documents/metro project/grid_proc.gpkg"

# Write the updated grid GeoDataFrame to file
grid_gdf.to_file(output_fp, driver="GPKG")