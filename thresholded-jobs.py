import geopandas as gpd

# Read grid and jobs GeoDataFrames
grid_gdf = gpd.read_file("C:/Users/Paul/Documents/metro project/county grid.gpkg")
jobs_gdf = gpd.read_file("C:/Users/Paul/Documents/metro project/jobs per census block.gpkg")

# Ensure both GeoDataFrames have the same CRS
if grid_gdf.crs != jobs_gdf.crs:
    jobs_gdf = jobs_gdf.to_crs(grid_gdf.crs)

# Initialize field to indicate which grid blocks are feasible
grid_gdf['IS_FEASIBLE'] = 0

# Define the radius of the catchment area (in the same units as the CRS, likely meters)
catchment_radius = 1800  # Example radius (1.8 km)

# Pre-specified threshold number of jobs
jobs_threshold = 1000  # Example threshold

# Get the centroids of job locations (to represent the job location)
jobs_gdf['centroid'] = jobs_gdf.geometry.centroid

# Loop through each grid block and calculate the total jobs within the catchment radius
for i, grid_row in grid_gdf.iterrows():
    # Get the centroid of the grid block
    grid_center = grid_row.geometry.centroid
    
    # Calculate distances from the grid center to each job location centroid
    jobs_gdf['distance'] = jobs_gdf['centroid'].distance(grid_center)
    
    # Filter jobs within the catchment radius
    within_catchment = jobs_gdf[jobs_gdf['distance'] <= catchment_radius]
    
    # Sum the jobs within the catchment area (assuming 'c000' column represents job counts)
    total_jobs = within_catchment['JOBS18'].sum()

    # Check if the total jobs meet the threshold
    if total_jobs >= jobs_threshold:
        grid_gdf.at[i, 'IS_FEASIBLE'] = 1

# Create an output path for the data
output_fp = "C:/Users/Paul/Documents/metro project/thresholded jobs grid.gpkg"

# Write the updated grid GeoDataFrame to file
grid_gdf.to_file(output_fp, driver="GPKG")