import os
import geopandas as gpd

#Get a GeoDataFrame of the grid
grid_gdf = gpd.read_file("C:\\Users\\Paul\\Documents\\metro project\\county grid.gpkg")
water_gdf = gpd.read_file("C:\\Users\\Paul\\Documents\\metro project\\water.gpkg")
if grid_gdf.crs != water_gdf.crs:
    water_gdf = water_gdf.to_crs(grid_gdf.crs)

#Area of each grid in meters
grid_area = 62500

intersection_gdf = gpd.overlay(grid_gdf, water_gdf, how="intersection")

# Calculate the area of the intersected geometries
intersection_gdf['water_area'] = intersection_gdf.geometry.area

water_area_per_grid = intersection_gdf.groupby('id')['water_area'].sum().reset_index()

grid_gdf = grid_gdf.merge(water_area_per_grid, on="id", how="left")

# Fill NaN values in water_area with 0 (if some grid cells don't intersect water)
grid_gdf['water_area'] = grid_gdf['water_area'].fillna(0)

# Set the water coverage percentage threshold (e.g., 30%)
threshold = 30

# Calculate the percentage of the grid covered by water
grid_gdf['water_percentage'] = (grid_gdf['water_area'] / grid_area) * 100

# Set the CONTAINS_WATER flag based on the threshold
grid_gdf['IS_FEASIBLE'] = (grid_gdf['water_percentage'] > threshold).astype(int)

# Reverse the IS_FEASIBLE values
grid_gdf['IS_FEASIBLE'] = 1 - grid_gdf['IS_FEASIBLE']

print(grid_gdf)

# Create a output path for the data
output_fp = "C:\\Users\\Paul\\Documents\\metro project\\water grid.gpkg"

# Write the file
grid_gdf.to_file(output_fp, driver="GPKG")

print("DONE PROCESSING")