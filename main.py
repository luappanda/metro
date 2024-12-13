import subprocess
import os
import time
from datetime import timedelta
import geopandas as gpd
import rasterio
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
# from rasterio.mask import mask
# from rasterio.plot import show


RED = '\033[31m'
GREEN = '\033[32m'
YELLOW = '\033[33m'
BLUE = '\033[34m'
RESET = '\033[0m'

# subprocess.run(["python", "primalgo.py"])
out_folder = os.getcwd() + "/GISFiles/output"
# # Create the Output Directory if It Doesn't Exist
# os.makedirs((out_folder), exist_ok=True)

runs = 10

total_time=0

# for x in range(runs):
#     os.makedirs((out_folder), exist_ok=True)
#     dir = os.listdir(out_folder)
#     if len(dir) > 0:
#         print('Please clear output directories')
#         exit()
#     start = time.time()
#     subprocess.run(["python", "corridor.py"])
#     subprocess.run(["python", "genetic-selection.py"])
#     subprocess.run(["python", "corridor2.py"])
#     subprocess.run(["python", "genetic-selection-line2.py"])
#     subprocess.run(["python", "corridor3.py"])
#     subprocess.run(["python", "genetic-selection-line3.py"])
#     subprocess.run(["python", "fitness-graph.py"])
#     subprocess.run(["python", "linealgo.py"])
#     subprocess.run(["python", "make-corridor-plot.py"])
#     subprocess.run(["python", "make-lines-plot.py"])
#     end = time.time()
#     elapsed_time = int(end-start)
#     estimated_time = elapsed_time*(runs-1-x)
#     total_time+=elapsed_time
#     print(f"{YELLOW}Elapsed Time: {str(timedelta(seconds=total_time))}")
#     print(f"{YELLOW}Estimated remaining Time: {str(timedelta(seconds=estimated_time))}{RESET}")
#     os.rename(out_folder, out_folder+str(x+1))



grid_filepath = os.getcwd() + "/GISFiles/weighted grid.gpkg"  # Update the path if necessary
grid_gdf = gpd.read_file(grid_filepath)
viable_grids = grid_gdf[grid_gdf["TOTAL WEIGHTED FEASIBILITY"] > 0].reset_index(drop=True)
population_filepath = os.getcwd() + "/GISFiles/block population.gpkg"
population_gdf = gpd.read_file(population_filepath)
population_gdf = population_gdf[~population_gdf.is_empty]  # Remove empty geometries
population_gdf = population_gdf[population_gdf.is_valid] 
if population_gdf.crs.is_geographic:
    population_gdf = population_gdf.to_crs("EPSG:3857")  # Use Web Mercator or appropriate CRS

# Normalize covered population as a score
max_population = population_gdf["POP20"].sum()

# Load the raster
raster_filepath = os.getcwd() + "/GISFiles/output_population_raster2.tif"  # Update the path if necessary
with rasterio.open(raster_filepath) as src:
    raster_crs = src.crs
    population_raster = src.read(1)  # Read population data (first band)
    transform = src.transform
    width = src.width
    height = src.height
    resolution = transform[0]  # Get resolution (pixel size in meters)



# 5. Constraints and Penalties
N_MIN = 5                  # Minimum number of stations
N_MAX = 10                # Maximum number of stations
D_MIN = 1800              # Minimum distance between stations in meters
D_MAX = 20000              # Maximum distance between stations in meters
POPULATION_RADIUS = 3000 # Radius around each station to consider population
MINIMUM__RSQUARED = 0.85  # Minimum R^2 value for the linear regression 
# Scaling factors for exponential penalties
ALPHA = 0.001              # Adjusted scaling factor for distance penalties
BETA = 1                 # Adjusted scaling factor for number of stations penalty

N_MIN *=3
N_MAX *=3

# Desired number of stations (set as the average of N_MIN and N_MAX)
N_DESIRED = (N_MIN + N_MAX) // 2


# Weights for the fitness function components
W1 = 9.0  # Adjusted weight for feasibility score
W2 = 3.0  # Increased weight for distance penalty
W3 = 0.0  # Increased weight for station count penalty
W4 = 2 # Weight for linearity
W5 = 10.0 # Weight for population coverage

def evaluate(individual):
    """
    Redesigned fitness function to balance feasibility, constraints, and linearity.
    """

    # Retrieve station geometries and normalized feasibility scores
    stations = individual.to_crs(raster_crs)
    feasibility_scores = stations["TOTAL WEIGHTED FEASIBILITY"].values
    total_feasibility = feasibility_scores.mean()


    # Calculate population coverage for each station
    stations_coords = []
    for station in stations.geometry:
        station_coords = (station.centroid.x, station.centroid.y)
        stations_coords.append(station_coords)

    resolution = transform[0]  # Pixel size (e.g., in meters per pixel)
    covered_population = calculate_population_in_radius(
            stations_coords, population_raster, transform, radius=POPULATION_RADIUS, width=width, height=height, resolution=resolution)

    # Sum population of intersecting features
    # covered_population = intersecting_population["POP20"].sum()
    # print(covered_population)

    population_score = covered_population / max_population

    # Distance Penalty
    distance_penalty = 0.0
    coords = stations.geometry.centroid.apply(lambda point: (point.x, point.y)).tolist()
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            d = np.linalg.norm(np.array(coords[i]) - np.array(coords[j]))
            if d < D_MIN:
                distance_penalty += np.exp(-ALPHA * (d - D_MIN))
            elif d > D_MAX:
                distance_penalty += np.exp(ALPHA * (d - D_MAX))
    # Normalize distance penalty
    max_possible_pairs = len(coords) * (len(coords) - 1) / 2
    if max_possible_pairs > 0:
        distance_penalty /= max_possible_pairs


    # Station Count Penalty
    N = len(individual)
    station_count_penalty = np.exp(BETA * abs(N - N_DESIRED))

    # Total Fitness Calculation
    fitness = (
        W1 * total_feasibility
        - W2 * distance_penalty
        - W3 * station_count_penalty
        + W5 * population_score
    )
    return (fitness, covered_population)


best_fitness = -np.inf
best_layout = None
best_population = None

def calculate_population_in_radius(station_coords, population_raster, transform, radius, width, height, resolution):
    # Convert station coordinates to pixel coordinates (Inverse transform)
    station_pixel_coords = [
        ~transform * (x, y)  # Apply inverse transform to (x, y) for pixel coordinates
        for (x, y) in station_coords
    ]

    # Convert radius to pixels
    radius_pixels = int(radius / resolution)  # Convert meters to pixels
    radius_squared = radius_pixels**2

    population_within_radius = 0
    visited = np.zeros((height, width), dtype=bool)  # Create a 2D array for visited pixels

    for station_pixel in station_pixel_coords:
        col, row = map(int, station_pixel)

        # Define the bounding box for the search area
        min_row = max(0, row - radius_pixels)
        max_row = min(height, row + radius_pixels + 1)
        min_col = max(0, col - radius_pixels)
        max_col = min(width, col + radius_pixels + 1)

        # Directly iterate over the bounding box
        for r in range(min_row, max_row):
            for c in range(min_col, max_col):
                dist_squared = (r - row)**2 + (c - col)**2
                if dist_squared <= radius_squared and not visited[r, c]:
                    population_within_radius += population_raster[r, c]
                    visited[r, c] = True  # Mark pixel as visited

    return population_within_radius



for i in range(runs):
    # Load stations
    out = "/output" + str(i+1)
    stations1_filepath = os.getcwd() + "/GISFiles"+out+"/best stations4.gpkg"
    stations1_gdf = gpd.read_file(stations1_filepath)
    stations2_filepath = os.getcwd() + "/GISFiles"+out+"/best stations2.gpkg"
    stations2_gdf = gpd.read_file(stations2_filepath)
    stations3_filepath = os.getcwd() + "/GISFiles"+out+"/best stations3.gpkg"
    stations3_gdf = gpd.read_file(stations3_filepath)
    total_stations = pd.concat([stations1_gdf, stations2_gdf]).drop_duplicates()
    total_stations = pd.concat([total_stations, stations3_gdf]).drop_duplicates()

    coords1 = stations1_gdf.geometry.centroid.apply(lambda point: (point.x, point.y)).tolist()
    coords2 = stations2_gdf.geometry.centroid.apply(lambda point: (point.x, point.y)).tolist()
    coords3 = stations3_gdf.geometry.centroid.apply(lambda point: (point.x, point.y)).tolist()
    X1 = np.array([c[0] for c in coords1]).reshape(-1, 1)  # x-coordinates
    y1 = np.array([c[1] for c in coords1])  # y-coordinates
    X2 = np.array([c[0] for c in coords2]).reshape(-1, 1)  # x-coordinates
    y2 = np.array([c[1] for c in coords2])  # y-coordinates
    X3 = np.array([c[0] for c in coords3]).reshape(-1, 1)  # x-coordinates
    y3 = np.array([c[1] for c in coords3])  # y-coordinates


    reg1 = LinearRegression().fit(X1, y1)
    reg2 = LinearRegression().fit(X2, y2)
    reg3 = LinearRegression().fit(X3, y3)
    r_squared1 = reg1.score(X1, y1)  # Coefficient of determination (R^2)
    r_squared2 = reg2.score(X2, y2)  # Coefficient of determination (R^2)
    r_squared3 = reg3.score(X3, y3)  # Coefficient of determination (R^2)

    linearity = (r_squared1+r_squared2+r_squared3)/3


    ids = total_stations["id"]
    stations = viable_grids.loc[viable_grids['id'].isin(ids)]
    fitness, covered_population = evaluate(stations) 
    fitness += W4 * linearity
    print(i+1, fitness, covered_population)
    if fitness > best_fitness:
        best_fitness = fitness
        best_layout = i+1
        best_population = covered_population

print("\nBest Fitness:", best_fitness)
print("\nBest Layout:", best_layout)
print("\nBest Population:", best_population)