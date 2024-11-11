import geopandas as gpd

# Load the grid files
jobs_gdf = gpd.read_file("C:/Users/Paul/Documents/metro project/weighted thresholded jobs grid.gpkg")
population_gdf = gpd.read_file("C:/Users/Paul/Documents/metro project/weighted thresholded walking distance grid.gpkg")
traffic_gdf = gpd.read_file("C:/Users/Paul/Documents/metro project/weighted traffic grid.gpkg")
water_gdf = gpd.read_file("C:/Users/Paul/Documents/metro project/water grid.gpkg")
bus_gdf = gpd.read_file("C:/Users/Paul/Documents/metro project/weighted bus station grid.gpkg")

grid_gdf = gpd.read_file("C:/Users/Paul/Documents/metro project/county grid.gpkg")

# Rename columns for clarity
jobs_gdf = jobs_gdf.rename(columns={'WEIGHTED_FEASIBILITY': 'JOBS_FEASIBILITY'})
population_gdf = population_gdf.rename(columns={'WEIGHTED_FEASIBILITY': 'POPULATION_FEASIBILITY'})
traffic_gdf = traffic_gdf.rename(columns={'WEIGHTED_FEASIBILITY': 'TRAFFIC_FEASIBILITY'})
water_gdf = water_gdf.rename(columns={'IS_FEASIBLE': 'WATER_FEASIBILITY'})
bus_gdf = bus_gdf.rename(columns={'WEIGHTED_FEASIBILITY': 'BUS_FEASIBILITY'})

# Merge all feasibility columns with the grid, based on geometry
combined_gdf = grid_gdf.merge(
    jobs_gdf[['geometry', 'JOBS_FEASIBILITY']], on='geometry', how='left'
).merge(
    population_gdf[['geometry', 'POPULATION_FEASIBILITY']], on='geometry', how='left'
).merge(
    traffic_gdf[['geometry', 'TRAFFIC_FEASIBILITY']], on='geometry', how='left'
).merge(
    water_gdf[['geometry', 'WATER_FEASIBILITY']], on='geometry', how='left'
).merge(
    bus_gdf[['geometry', 'BUS_FEASIBILITY']], on='geometry', how='left'
)

# Initialize the TOTAL WEIGHTED FEASIBILITY column
combined_gdf['TOTAL WEIGHTED FEASIBILITY'] = 0

# Calculate total feasibility with Python logic
for idx, row in combined_gdf.iterrows():
    feasibilities = [
        row['JOBS_FEASIBILITY'],
        row['POPULATION_FEASIBILITY'],
        row['TRAFFIC_FEASIBILITY'],
        row['BUS_FEASIBILITY']
    ]
    if any(f == 0 for f in feasibilities) or combined_gdf.at[idx, 'WATER_FEASIBILITY'] == 0:
        combined_gdf.at[idx, 'TOTAL WEIGHTED FEASIBILITY'] = 0
    else:
        combined_gdf.at[idx, 'TOTAL WEIGHTED FEASIBILITY'] = sum(feasibilities) / len(feasibilities)

# Update the original grid with the combined feasibility values
grid_gdf['TOTAL WEIGHTED FEASIBILITY'] = combined_gdf['TOTAL WEIGHTED FEASIBILITY']

# Save the result to a new file
output_fp = "C:/Users/Paul/Documents/metro project/weighted grid.gpkg"
grid_gdf.to_file(output_fp, driver="GPKG")