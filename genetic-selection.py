import os
import geopandas as gpd
import numpy as np
import random
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt

# ------------------------------ #
#       Genetic Algorithm Setup   #
# ------------------------------ #

# 1. Load the Weighted Feasibility Grid
grid_filepath = "C:/Users/kavan_3rgiqdq/Documents/metro project/weighted grid.gpkg"  # Update the path if necessary
grid_gdf = gpd.read_file(grid_filepath)

# 2. Filter Viable Grids Based on Feasibility
viable_grids = grid_gdf[grid_gdf["TOTAL WEIGHTED FEASIBILITY"] > 0].reset_index(drop=True)

# Ensure the GeoDataFrame is projected for accurate distance calculations
# Replace 'EPSG:YOUR_UTM_ZONE' with the appropriate EPSG code for your area
# For example, EPSG:32615 for UTM zone 15N
# If already projected, you can skip this step
# viable_grids = viable_grids.to_crs(epsg=32615)

# 3. Normalize Weights for Selection Probability
weights = viable_grids["TOTAL WEIGHTED FEASIBILITY"].values
weights_normalized = weights / weights.sum()

# 4. Genetic Algorithm Parameters
NUM_STATIONS = 15         # Number of stations per individual
POPULATION_SIZE = 40      # Number of individuals in the population
NUM_GENERATIONS = 600    # Number of generations
CX_PROB = 0.5             # Crossover probability
MUT_PROB = 0.2            # Mutation probability
SEED = 23                 # Random seed for reproducibility

# 5. Constraints and Penalties
D_MIN = 5000               # Minimum distance between stations in meters
D_MAX = 20000              # Maximum distance between stations in meters
N_MIN = 5                  # Minimum number of stations
N_MAX = 15                 # Maximum number of stations
P_CLOSE = 100              # Penalty for stations too close
P_FAR = 50                 # Penalty for stations too far
P_N = 1000                 # Penalty for violating number of stations
# Terminal penalties can be added if terminal data is available

# 6. Initialize Random Seed
random.seed(SEED)
np.random.seed(SEED)

# 7. Function to Select Initial Stations Based on Weighted Feasibility
def select_initial_stations():
    selected_indices = np.random.choice(
        viable_grids.index,
        size=NUM_STATIONS,
        p=weights_normalized,
        replace=False
    )
    return list(selected_indices)

# 8. Define the Fitness Function
def evaluate(individual):
    """
    Fitness function to evaluate the quality of station placement.
    The goal is to maximize the total weighted feasibility while minimizing penalties.
    """
    # Retrieve station geometries
    stations = viable_grids.loc[individual]
    
    # Total Feasibility Score
    total_feasibility = stations["TOTAL WEIGHTED FEASIBILITY"].sum()
    
    # Initialize Penalties
    distance_penalty = 0
    num_stations_penalty = 0
    
    # Number of Stations Penalty
    N = len(individual)
    if N < N_MIN or N > N_MAX:
        num_stations_penalty = P_N
    
    # Calculate Distance Penalties Between Stations
    # Extract coordinates for distance calculations
    coords = stations.geometry.centroid.apply(lambda point: (point.x, point.y)).tolist()
    
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            d_nm = np.linalg.norm(np.array(coords[i]) - np.array(coords[j]))
            if d_nm < D_MIN:
                distance_penalty += P_CLOSE
            elif d_nm > D_MAX:
                distance_penalty += P_FAR
    
    # Total Fitness Calculation
    fitness = total_feasibility - distance_penalty - num_stations_penalty
    return (fitness,)

# 9. Set Up DEAP Framework
# Define the Fitness and Individual Classes
creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Maximizing fitness
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Attribute Generator: Select a random station index from viable_grids
toolbox.register("attr_station", random.choice, viable_grids.index.tolist())

# Structure Initializers
# Initialize individuals as lists of station indices
def init_individual():
    return list(np.random.choice(
        viable_grids.index,
        size=NUM_STATIONS,
        p=weights_normalized,
        replace=False
    ))

toolbox.register("individual", tools.initIterate, creator.Individual, init_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Register the Evaluation Function
toolbox.register("evaluate", evaluate)

# Register Genetic Operators
toolbox.register("mate", tools.cxTwoPoint)

# Custom Mutation Operator to Ensure Unique Stations
def mutate_unique(individual, indpb):
    """
    Mutates an individual by replacing some of its genes with new unique genes.
    """
    for i in range(len(individual)):
        if random.random() < indpb:
            new_gene = random.choice(viable_grids.index.tolist())
            while new_gene in individual:
                new_gene = random.choice(viable_grids.index.tolist())
            individual[i] = new_gene
    return (individual,)

toolbox.register("mutate", mutate_unique, indpb=0.2)

# Selection Operator: Tournament Selection
toolbox.register("select", tools.selTournament, tournsize=3)

# 10. Prevent Duplicate Stations in Individuals
# Ensure that individuals have unique stations
def check_unique(individual):
    return len(individual) == len(set(individual))

# 11. Main Genetic Algorithm Function
def main():
    # Initialize Population
    population = toolbox.population(n=POPULATION_SIZE)
    
    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
    
    # Statistics to Keep Track of Progress
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    
    # Hall of Fame to Store the Best Individuals
    hof = tools.HallOfFame(1)
    
    # Genetic Algorithm Parameters
    pop, log = algorithms.eaSimple(
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
    
    # Retrieve the Best Stations GeoDataFrame
    best_stations = viable_grids.loc[best_individual]
    
    # Output Path for the Best Stations
    output_fp = "C:/Users/kavan_3rgiqdq/Documents/metro project/best stations.gpkg"
    
    # Create the Output Directory if It Doesn't Exist
    os.makedirs(os.path.dirname(output_fp), exist_ok=True)
    
    # Save the Best Stations to a GeoPackage
    best_stations.to_file(output_fp, driver="GPKG")
    print(f"Best stations saved to {output_fp}")
    
    # Optional: Visualize the Best Stations
    visualize_results(viable_grids, best_stations)

# 12. Visualization Function (Optional)
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

# 13. Run the Genetic Algorithm
if __name__ == "__main__":
    main()