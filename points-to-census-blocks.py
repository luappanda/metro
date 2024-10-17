import geopandas as gpd

# Read grid and population GeoDataFrames
blocks_gdf = gpd.read_file("C:/Users/Paul/Documents/metro project/block population.gpkg")
jobs_gdf = gpd.read_file("C:/Users/Paul/Documents/metro project/jobs.gpkg")

# Ensure both GeoDataFrames have the same CRS
if blocks_gdf.crs != jobs_gdf.crs:
    jobs_gdf = jobs_gdf.to_crs(blocks_gdf.crs)

# Initialize field to indicate which grid blocks are feasible
blocks_gdf['JOBS18'] = 0

intersection_gdf = gpd.sjoin(blocks_gdf, jobs_gdf, how="left", predicate="intersects")

# Filter out rows from intersection_gdf where 'index_right' is not null (meaning water is intersecting)
# Get the unique indices from the grid_gdf where water is present
water_indices = intersection_gdf[intersection_gdf['index_right'].notnull()].index.unique()

# Set 'CONTAINS_WATER' to 1 for rows in grid_gdf that have an index in water_indices
grid_gdf.loc[water_indices, 'CONTAINS_WATER'] = 1

# Create an output path for the data
output_fp = "C:/Users/Paul/Documents/metro project/jobs per census block.gpkg"

# Write the updated grid GeoDataFrame to file
blocks_gdf.to_file(output_fp, driver="GPKG")