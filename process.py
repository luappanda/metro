import os
import geopandas as gpd

#Get a GeoDataFrame of the grid
grid_gdf = gpd.read_file("GISFiles/CountyGrid.gpkg")
water_gdf = gpd.read_file("GISFiles/water.gpkg")
if grid_gdf.crs != water_gdf.crs:
    water_gdf = water_gdf.to_crs(grid_gdf.crs)

grid_gdf['CONTAINS_WATER'] = 0  # initialize field, float, two decimals

intersection_gdf = gpd.sjoin(grid_gdf, water_gdf, how="left", predicate="intersects")

# Filter out rows from intersection_gdf where 'index_right' is not null (meaning water is intersecting)
# Get the unique indices from the grid_gdf where water is present
water_indices = intersection_gdf[intersection_gdf['index_right'].notnull()].index.unique()

# Set 'CONTAINS_WATER' to 1 for rows in grid_gdf that have an index in water_indices
grid_gdf.loc[water_indices, 'CONTAINS_WATER'] = 1


# Create a output path for the data
output_fp = os.getcwd() + "/GISFiles/grid_proc.gpkg"

# Write the file
grid_gdf.to_file(output_fp, driver="GPKG")