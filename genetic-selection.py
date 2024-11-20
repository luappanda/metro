import os
import geopandas as gpd
import numpy as np
import random
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt

# ----------------------------------- #
#       Genetic Algoritfhm Setup       #
# ----------------------------------- #

# 1. Load the Weighted Feasibility Grid
grid_filepath = os.getcwd() + "/GISFiles/weighted grid.gpkg"  # Update the path if necessary
grid_gdf = gpd.read_file(grid_filepath)

# 2. Filter Viable Grids Based on Feasibility
viable_grids = grid_gdf[grid_gdf["TOTAL WEIGHTED FEASIBILITY"] > 0].reset_index(drop=True)

# 3. Normalize Weights for Selection Probability
weights = viable_grids["TOTAL WEIGHTED FEASIBILITY"].values
weights_normalized = weights / weights.sum()

# 4. Genetic Algorithm Parameters
N_MIN = 10                 # Minimum number of stations
N_MAX = 30                 # Maximum number of stations
POPULATION_SIZE = 30       # Number of individuals in the population
NUM_GENERATIONS = 200      # Number of generations
CX_PROB = 0.5              # Crossover probability
MUT_PROB = 0.35            # Mutation probability
SEED = 37                  # Random seed for reproducibility

# 5. Constraints and Penalties
D_MIN = 5000               # Minimum distance between stations in meters
D_MAX = 35000              # Maximum distance between stations in meters
P_CLOSE = 2000             # Penalty factor for stations too close
P_FAR = 50                 # Penalty factor for stations too far
P_N = 1000                 # Penalty factor for violating number of stations

# Scaling factors for exponential penalties
ALPHA = 0.001              # Scaling factor for distance penalties
BETA = 0.5                 # Scaling factor for number of stations penalty

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

# ... [Previous code remains unchanged up to step 8]

# Precompute normalized feasibility scores
min_feasibility = viable_grids["TOTAL WEIGHTED FEASIBILITY"].min()
max_feasibility = viable_grids["TOTAL WEIGHTED FEASIBILITY"].max()
viable_grids["Normalized Feasibility"] = (
    viable_grids["TOTAL WEIGHTED FEASIBILITY"] - min_feasibility
) / (max_feasibility - min_feasibility)

# Desired number of stations (set as the average of N_MIN and N_MAX)
N_DESIRED = (N_MIN + N_MAX) // 2

# Weights for the fitness function components
W1 = 2.0  # Weight for feasibility score
W2 = 3.0  # Weight for distance penalty
W3 = 1.0  # Weight for station count penalty

def evaluate(individual):
    """
    Redesigned fitness function to balance feasibility and constraints.
    """
    # Retrieve station geometries and normalized feasibility scores
    stations = viable_grids.loc[individual]
    feasibility_scores = stations["Normalized Feasibility"].values
    
    # Total Normalized Feasibility Score
    total_feasibility = feasibility_scores.sum() / N_DESIRED  # Normalize by desired number of stations
    
    # Distance Penalty
    distance_penalty = 0.0
    coords = stations.geometry.centroid.apply(lambda point: (point.x, point.y)).tolist()
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            d = np.linalg.norm(np.array(coords[i]) - np.array(coords[j]))
            if d < D_MIN:
                distance_penalty += ( (D_MIN - d) / D_MIN ) ** 2
            elif d > D_MAX:
                distance_penalty += ( (d - D_MAX) / D_MAX ) ** 2
    # Normalize distance penalty
    max_possible_pairs = len(coords) * (len(coords) - 1) / 2
    if max_possible_pairs > 0:
        distance_penalty /= max_possible_pairs
    
    # Station Count Penalty
    N = len(individual)
    station_count_penalty = ( (N - N_DESIRED) / N_DESIRED ) ** 2
    
    # Total Fitness Calculation
    fitness = (
        W1 * total_feasibility
        - W2 * distance_penalty
        - W3 * station_count_penalty
    )
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
            ind1[ind1.index(gene1)] = gene2
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

toolbox.register("mutate", mutate_individual, indpb=0.2)

# Selection Operator: Tournament Selection
toolbox.register("select", tools.selTournament, tournsize=3)

# 10. Main Genetic Algorithm Function
# 10. Main Genetic Algorithm Function
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
