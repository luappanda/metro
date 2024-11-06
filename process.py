import geopandas as gpd

# Load the grid files
jobs_gdf = gpd.read_file("C:/Users/Paul/Documents/metro project/weighted thresholded jobs grid.gpkg")
population_gdf = gpd.read_file("C:/Users/Paul/Documents/metro project/c:\Users\Paul\Documents\metro project\weighted thresholded walking distance grid.gpkg")
traffic_gdf = gpd.read_file("C:/Users/Paul/Documents/metro project/weighted traffic grid.gpkg")
water_gdf = gpd.read_file("C:/Users/Paul/Documents/metro project/water.gpkg")
bus_gdf = gpd.read_file("C:/Users/Paul/Documents/metro project/water.gpkg")

grid_gdf = gpd.read_file("C:/Users/Paul/Documents/metro project/county grid.gpkg")

# Rename columns for clarity
jobs_gdf = jobs_gdf.rename(columns={'WEIGHTED_FEASIBILITY': 'JOBS_FEASIBILITY'})
traffic_gdf = traffic_gdf.rename(columns={'WEIGHTED_FEASIBILITY': 'TRAFFIC_FEASIBILITY'})

# Merge by geometry
combined_gdf = grid_gdf.merge(
    jobs_gdf[['geometry', 'JOBS_FEASIBILITY']], on='geometry', how='left'
).merge(
    traffic_gdf[['geometry', 'TRAFFIC_FEASIBILITY']], on='geometry', how='left'
)

# Fill NaNs with 0 for missing feasibility values
combined_gdf['JOBS_FEASIBILITY'] = combined_gdf['JOBS_FEASIBILITY'].fillna(0)
combined_gdf['TRAFFIC_FEASIBILITY'] = combined_gdf['TRAFFIC_FEASIBILITY'].fillna(0)

# Initialize the TOTAL WEIGHTED FEASIBILITY column
combined_gdf['TOTAL WEIGHTED FEASIBILITY'] = 0

# Calculate total feasibility with plain Python logic
for idx, row in combined_gdf.iterrows():
    if row['JOBS_FEASIBILITY'] == 0 or row['TRAFFIC_FEASIBILITY'] == 0:
        combined_gdf.at[idx, 'TOTAL WEIGHTED FEASIBILITY'] = 0
    else:
        combined_gdf.at[idx, 'TOTAL WEIGHTED FEASIBILITY'] = (row['JOBS_FEASIBILITY'] + row['TRAFFIC_FEASIBILITY']) / 2

# Update the original grid with the combined feasibility values
grid_gdf['TOTAL WEIGHTED FEASIBILITY'] = combined_gdf['TOTAL WEIGHTED FEASIBILITY']

# Save the result to a new file
output_fp = "C:/Users/Paul/Documents/metro project/weighted grid.gpkg"
grid_gdf.to_file(output_fp, driver="GPKG")
