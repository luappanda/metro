import os
import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree
import random
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from shapely.ops import unary_union
import rasterio
from rasterio.mask import mask
from rasterio.plot import show
from shapely.geometry import Point
from affine import Affine
from joblib import Parallel, delayed


# ----------------------------------- #
#       Genetic Algoritfhm Setup       #
# ----------------------------------- #

# Global cache for fitness evaluations
fitness_cache = {}

# 1. Load the Weighted Feasibility Grid
grid_filepath = os.getcwd() + "/GISFiles/weighted grid.gpkg"  # Update the path if necessary
grid_gdf = gpd.read_file(grid_filepath)
population_filepath = os.getcwd() + "/GISFiles/block population.gpkg"
population_gdf = gpd.read_file(population_filepath)
population_gdf = population_gdf[~population_gdf.is_empty]  # Remove empty geometries
population_gdf = population_gdf[population_gdf.is_valid] 
if population_gdf.crs.is_geographic:
    population_gdf = population_gdf.to_crs("EPSG:3857")  # Use Web Mercator or appropriate CRS

# Load the raster
raster_filepath = os.getcwd() + "/GISFiles/output_population_raster2.tif"  # Update the path if necessary
with rasterio.open(raster_filepath) as src:
    raster_crs = src.crs
    population_raster = src.read(1)  # Read population data (first band)
    transform = src.transform
    width = src.width
    height = src.height
    resolution = transform[0]  # Get resolution (pixel size in meters)

# 2. Filter Viable Grids Based on Feasibility
viable_grids = grid_gdf[grid_gdf["TOTAL WEIGHTED FEASIBILITY"] > 0].reset_index(drop=True)

# 3. Normalize Weights for Selection Probability
weights = viable_grids["TOTAL WEIGHTED FEASIBILITY"].values
weights_normalized = weights / weights.sum()

# Normalize covered population as a score
max_population = population_gdf["POP20"].sum()

# 4. Genetic Algorithm Parameters
N_MIN = 5                  # Minimum number of stations
N_MAX = 10                # Maximum number of stations
POPULATION_SIZE = 150      # Increased population size
NUM_GENERATIONS = 650      # Increased number of generations
CX_PROB = 0.75              # Increased crossover probability
MUT_PROB = 0.35             # Increased mutation probability
SEED = 23                  # Random seed for reproducibility

# 5. Constraints and Penalties
D_MIN = 2000              # Minimum distance between stations in meters
D_MAX = 20000              # Maximum distance between stations in meters
POPULATION_RADIUS = 2000 # Radius around each station to consider population

# Scaling factors for exponential penalties
ALPHA = 0.001              # Adjusted scaling factor for distance penalties
BETA = 1.0                 # Adjusted scaling factor for number of stations penalty

# 6. Initialize Random Seed
random.seed(SEED)
np.random.seed(SEED)

# 7. Function to Select Initial Stations Based on Weighted Feasibility
def init_individual():
    N = random.randint(N_MIN, N_MAX)
    return list(np.random.choice(
        viable_grids.index,
        size=N,
        p=weights_normalized,
        replace=False
    ))

# Precompute normalized feasibility scores
min_feasibility = viable_grids["TOTAL WEIGHTED FEASIBILITY"].min()
max_feasibility = viable_grids["TOTAL WEIGHTED FEASIBILITY"].max()
viable_grids["Normalized Feasibility"] = (
    viable_grids["TOTAL WEIGHTED FEASIBILITY"] - min_feasibility
) / (max_feasibility - min_feasibility)

# Desired number of stations (set as the average of N_MIN and N_MAX)
N_DESIRED = (N_MIN + N_MAX) // 2

# Weights for the fitness function components
W1 = 9.0  # Adjusted weight for feasibility score
W2 = 1.0  # Increased weight for distance penalty
W3 = 3.0  # Increased weight for station count penalty
W4 = 15.0 # Weight for linearity
W5 = 4.0 # Weight for population coverage

def evaluate(individual):
    """
    Redesigned fitness function to balance feasibility, constraints, and linearity.
    """
    # Create a hashable key from the individual
    key = tuple(sorted(individual))
    if key in fitness_cache:
        # Return cached fitness value
        return (fitness_cache[key],)

    # Retrieve station geometries and normalized feasibility scores
    stations = viable_grids.loc[individual]
    stations = stations.to_crs(raster_crs)
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
    # print(population_score)


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

    # Linearity Penalty/Reward
    if len(coords) > 1:
        X = np.array([c[0] for c in coords]).reshape(-1, 1)  # x-coordinates
        y = np.array([c[1] for c in coords])  # y-coordinates
        reg = LinearRegression().fit(X, y)
        r_squared = reg.score(X, y)  # Coefficient of determination (R^2)
        linearity_score = 1 - r_squared  # Penalize deviation from perfect linearity
    else:
        linearity_score = 1.0  # Maximum penalty for single station

    # Total Fitness Calculation
    fitness = (
        W1 * total_feasibility
        - W2 * distance_penalty
        - W3 * station_count_penalty
        - W4 * linearity_score
        + W5 * population_score
    )
    fitness_cache[key] = fitness
    return (fitness,)

# 9. Set Up DEAP Framework
creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Maximizing fitness
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Structure Initializers
toolbox.register("individual", tools.initIterate, creator.Individual, init_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Register the Evaluation Function
toolbox.register("evaluate", evaluate)

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


# Custom Crossover Operator for Variable-Length Individuals
def crossover_individuals(ind1, ind2):
    """
    Perform crossover between two individuals with potentially different lengths.
    """
    # Convert to sets to find common and unique genes
    set1, set2 = set(ind1), set(ind2)
    common = list(set1 & set2)
    unique1 = list(set1 - set2)
    unique2 = list(set2 - set1)

    # Swap a random number of unique genes
    swap_size = min(len(unique1), len(unique2))
    if swap_size > 0:
        num_swaps = random.randint(1, swap_size)
        indices1 = random.sample(range(len(unique1)), num_swaps)
        indices2 = random.sample(range(len(unique2)), num_swaps)

        for idx1, idx2 in zip(indices1, indices2):
            gene1 = unique1[idx1]
            gene2 = unique2[idx2]
            if gene2 not in ind1:
                ind1[ind1.index(gene1)] = gene2
            if gene1 not in ind2:
                ind2[ind2.index(gene2)] = gene1

    return ind1, ind2

toolbox.register("mate", crossover_individuals)

# Custom Mutation Operator for Variable-Length Individuals
def mutate_individual(individual, indpb):
    """
    Mutate an individual by adding, deleting, or modifying stations.
    """
    actions = ['add', 'delete', 'modify']
    action = random.choice(actions)

    if action == 'add' and len(individual) < N_MAX:
        new_gene = random.choice(viable_grids.index.tolist())
        while new_gene in individual:
            new_gene = random.choice(viable_grids.index.tolist())
        individual.append(new_gene)
    elif action == 'delete' and len(individual) > N_MIN:
        del individual[random.randrange(len(individual))]
    elif action == 'modify':
        for i in range(len(individual)):
            if random.random() < indpb:
                new_gene = random.choice(viable_grids.index.tolist())
                while new_gene in individual:
                    new_gene = random.choice(viable_grids.index.tolist())
                individual[i] = new_gene
    return (individual,)

toolbox.register("mutate", mutate_individual, indpb=0.3)

# Selection Operator: Tournament Selection with lower pressure
toolbox.register("select", tools.selTournament, tournsize=2)

# 10. Main Genetic Algorithm Function
# 10. Main Genetic Algorithm Function
def main():
    # Initialize Population
    population = toolbox.population(n=POPULATION_SIZE)

    ## Parallelize the evaluation of the fitness function
    num_jobs = -1  # Use all available CPU cores, or set it to a specific number (e.g., 4)
    fitnesses = Parallel(n_jobs=num_jobs)(
        delayed(toolbox.evaluate)(ind) for ind in population
    )

    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # Statistics to Keep Track of Progress
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    # Hall of Fame to Store the Best Individuals
    hof = tools.HallOfFame(1)

    # Genetic Algorithm Parameters
    algorithms.eaSimple(
        population,
        toolbox,
        cxpb=CX_PROB,
        mutpb=MUT_PROB,
        ngen=NUM_GENERATIONS,
        stats=stats,
        halloffame=hof,
        verbose=True
    )

    # Retrieve the Best Individual
    best_individual = hof[0]
    print("Best Individual Fitness:", best_individual.fitness.values[0])
    print("Best Individual Stations Indices:", best_individual)

    # Calculate Penalties for Loss Row
    stations = viable_grids.loc[best_individual]
    feasibility_scores = stations["Normalized Feasibility"].values
    total_feasibility = feasibility_scores.sum() / N_DESIRED
    
    # Distance Penalty
    coords = stations.geometry.centroid.apply(lambda point: (point.x, point.y)).tolist()
    distance_penalty = 0.0
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            d = np.linalg.norm(np.array(coords[i]) - np.array(coords[j]))
            if d < D_MIN:
                distance_penalty += ((D_MIN - d) / D_MIN) ** 2
            elif d > D_MAX:
                distance_penalty += ((d - D_MAX) / D_MAX) ** 2
    max_possible_pairs = len(coords) * (len(coords) - 1) / 2
    if max_possible_pairs > 0:
        distance_penalty /= max_possible_pairs

    # Station Count Penalty
    N = len(best_individual)
    station_count_penalty = ((N - N_DESIRED) / N_DESIRED) ** 2

    # Loss Calculation
    total_loss = W2 * distance_penalty + W3 * station_count_penalty
    print(f"Loss Contribution: {total_loss:.4f}")

    # Retrieve the Best Stations GeoDataFrame
    best_stations = viable_grids.loc[best_individual]

    # Output Path for the Best Stations
    output_fp = os.getcwd() + "/GISFiles/best stations.gpkg"

    # Create the Output Directory if It Doesn't Exist
    os.makedirs(os.path.dirname(output_fp), exist_ok=True)

    # Save the Best Stations to a GeoPackage
    best_stations.to_file(output_fp, driver="GPKG")
    print(f"Best stations saved to {output_fp}")

    # Optional: Visualize the Best Stations
    visualize_results(viable_grids, best_stations)

# 11. Visualization Function (Optional)
def visualize_results(all_grids, best_stations):
    """
    Visualizes all viable grids and highlights the best station placements.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot All Viable Grids
    all_grids.plot(ax=ax, color='lightgray', edgecolor='white')

    # Plot Best Stations
    best_stations.plot(ax=ax, marker='o', color='red', markersize=50, label='Best Stations')

    # Add Legend and Title
    plt.legend()
    plt.title("Optimized Station Placements")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    # Show Plot
    plt.show()

# 12. Run the Genetic Algorithm
if __name__ == "__main__":
    main()
