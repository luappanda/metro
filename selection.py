import os
import geopandas as gpd
import random

#Get a GeoDataFrame of the grid
grid_gdf = gpd.read_file("GISFiles/grid_proc.gpkg")

viable_grids = grid_gdf[grid_gdf["CONTAINS_WATER"] == False]

selected_grids = viable_grids.sample(n=20)

# Create a output path for the data
output_fp = os.getcwd() + "/GISFiles/selection.gpkg"

# Write the file
selected_grids.to_file(output_fp, driver="GPKG")