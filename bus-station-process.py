import os
import geopandas as gpd

#Get a GeoDataFrame of the grid
grid_gdf = gpd.read_file("C:/Users/Paul/Documents/metro project/county grid.gpkg")
stations_gdf = gpd.read_file("C:/Users/Paul/Documents/metro project/bus stations.gpkg")

if grid_gdf.crs != stations_gdf.crs:
    stations_gdf = stations_gdf.to_crs(grid_gdf.crs)

grid_gdf['CONTAINS_STATION'] = 0  # initialize field, float, two decimals


# Create a output path for the data
output_fp = "C:/Users/Paul/Documents/metro project/grid_proc.gpkg"

# Write the file
grid_gdf.to_file(output_fp, driver="GPKG")