import geopandas as gpd

# Read grid and population GeoDataFrames
grid_gdf = gpd.read_file("C:/Users/Paul/Documents/metro project/county grid.gpkg")
population_gdf = gpd.read_file("C:/Users/Paul/Documents/metro project/block population.gpkg")

# Ensure both GeoDataFrames have the same CRS
if grid_gdf.crs != population_gdf.crs:
    population_gdf = population_gdf.to_crs(grid_gdf.crs)

# Initialize field to indicate which grid blocks are feasible
grid_gdf['IS_FEASIBLE'] = 0

# Define the radius of the catchment area (in the same units as the CRS, likely meters)
# Assuming a 5 minute driving time is the maximum, and the car travels at an average speed of 40 mph = about 5000 meters
catchment_radius = 1800  # Example: 5 km (R_p)

# Pre-specified threshold number of people (P_p)
population_threshold = 3000

# Get the centroids of population blocks (to represent the block location)
population_gdf['centroid'] = population_gdf.geometry.centroid

# Loop through each grid block and calculate the total population within the catchment radius
for i, grid_row in grid_gdf.iterrows():
    # Get the centroid of the grid block
    grid_center = grid_row.geometry.centroid
    
    # Calculate distances from the grid center to each population block centroid
    population_gdf['distance'] = population_gdf['centroid'].distance(grid_center)
    
    # Filter population blocks within the catchment radius
    within_catchment = population_gdf[population_gdf['distance'] <= catchment_radius]
    
    # Sum the population within the catchment area (assuming 'population' column exists in population_gdf)
    total_population = within_catchment['POP20'].sum()

    # Check if the total population meets the threshold P_p
    if total_population >= population_threshold:
        grid_gdf.at[i, 'IS_FEASIBLE'] = 1

# Create an output path for the data
output_fp = "C:/Users/Paul/Documents/metro project/thresholded walking distance grid.gpkg"

# Write the updated grid GeoDataFrame to file
grid_gdf.to_file(output_fp, driver="GPKG")