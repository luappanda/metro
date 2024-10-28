import geopandas as gpd

# Read grid and population GeoDataFrames
grid_gdf = gpd.read_file("C:/Users/Paul/Documents/metro project/thresholded jobs grid.gpkg")

grid_gdf['WEIGHTED_FEASIBILITY'] = 0

# Define decay function parameters
decay_constant = 100  # Decay constant for how fast feasibility fades, adjust based on the scale of your grid

# Get feasible blocks (where IS_FEASIBLE == 1)
feasible_blocks = grid_gdf[grid_gdf['IS_FEASIBLE'] == 1]

# Loop through each grid block
for i, grid_row in grid_gdf.iterrows():
    if grid_row['IS_FEASIBLE'] == 1:
        # If the block is already feasible, assign max weight (e.g., 1)
        grid_gdf.at[i, 'WEIGHTED_FEASIBILITY'] = 1
    else:
        # If the block is not feasible, calculate the distance to the nearest feasible block
        grid_center = grid_row.geometry.centroid
        
        # Calculate the distance to each feasible block's centroid
        feasible_blocks['distance'] = feasible_blocks.geometry.centroid.distance(grid_center)
        
        # Find the minimum distance to any feasible block
        min_distance = feasible_blocks['distance'].min()
        
        # Apply the decay function to calculate the weight based on distance
        weight = np.exp(-min_distance / decay_constant)
        
        # Assign the calculated weight to the grid block
        grid_gdf.at[i, 'WEIGHTED_FEASIBILITY'] = weight

# Create an output path for the data
output_fp = "C:/Users/Paul/Documents/metro project/weighted thresholded jobs grid.gpkg"

# Write the updated grid GeoDataFrame to file
grid_gdf.to_file(output_fp, driver="GPKG")