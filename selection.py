import os
import geopandas as gpd
import numpy as np
import random

#Get a GeoDataFrame of the grid
grid_gdf = gpd.read_file("GISFiles/weighted grid.gpkg")

viable_grids = grid_gdf[grid_gdf["TOTAL WEIGHTED FEASIBILITY"] > 0]
weights = viable_grids["TOTAL WEIGHTED FEASIBILITY"].values / viable_grids["TOTAL WEIGHTED FEASIBILITY"].sum()


num_samples = 10  # Number of points to select
selected_indices = np.random.choice(viable_grids.index, size=num_samples, p=weights, replace=False)
print(selected_indices)

selected_grids = viable_grids.loc[selected_indices]

# Create a output path for the data
output_fp = os.getcwd() + "/GISFiles/selection.gpkg"

# Write the file
selected_grids.to_file(output_fp, driver="GPKG")