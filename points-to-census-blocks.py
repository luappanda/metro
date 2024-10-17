import geopandas as gpd
import matplotlib.pyplot as plt

# Read grid and population GeoDataFrames
blocks_gdf = gpd.read_file("C:/Users/Paul/Documents/metro project/block population.gpkg")
jobs_gdf = gpd.read_file("C:/Users/Paul/Documents/metro project/jobss.gpkg")

# Reproject to a projected CRS for Arkansas (EPSG:26915)
projected_crs = "EPSG:26915"
blocks_gdf = blocks_gdf.to_crs(projected_crs)
jobs_gdf = jobs_gdf.to_crs(projected_crs)

# Initialize field to indicate which grid blocks are feasible
blocks_gdf['JOBS18'] = 0

# Get the centroids of job locations (to represent the job location)
jobs_centroids_gdf = gpd.GeoDataFrame(jobs_gdf.copy(), geometry=jobs_gdf.centroid, crs=projected_crs)

# Loop through each grid block and calculate the total jobs within it
for i, grid_row in blocks_gdf.iterrows():
    grid_gdf = gpd.GeoDataFrame([grid_row], crs=blocks_gdf.crs)

    # Perform the spatial join
    intersection_gdf = gpd.sjoin(grid_gdf, jobs_centroids_gdf, how="left", predicate="intersects")

    # Filter jobs within the current block
    jobs_indices = intersection_gdf[intersection_gdf['index_right'].notnull()]['index_right'].unique()
    within_block = jobs_gdf.loc[jobs_indices]

    # Sum the jobs in the current block
    blocks_gdf.at[i, 'JOBS18'] = within_block['c000'].sum()

# Create an output path for the data
output_fp = "C:/Users/Paul/Documents/metro project/jobs per census block.gpkg"

# Write the updated grid GeoDataFrame to file
blocks_gdf.to_file(output_fp, driver="GPKG")