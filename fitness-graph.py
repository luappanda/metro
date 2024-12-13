import matplotlib.pyplot as plt
import numpy as np
import json
import os

json_fp = os.getcwd() + "/GISFiles/output/fitness_data.json"

# Open and read the JSON file
with open(json_fp, 'r') as file:
    data = json.load(file)

avg_fitness1 = np.array(data["avg_fitness"])
avg_fitness2 = np.array(data["avg_fitness2"])
avg_fitness3 = np.array(data["avg_fitness3"])
max_fitness1 = np.array(data["max_fitness"])
max_fitness2 = np.array(data["max_fitness2"])
max_fitness3 = np.array(data["max_fitness3"])

gen = np.array(data["generations"])

def normalize_fitness(fitness_values):
    # Min-max normalization
    min_val = fitness_values.min()
    max_val = fitness_values.max()

    normalized_values = (fitness_values - min_val) / (max_val - min_val)
    return normalized_values

norm_fitness1 = normalize_fitness(avg_fitness1)
norm_fitness2 = normalize_fitness(avg_fitness2)
norm_fitness3 = normalize_fitness(avg_fitness3)

norm_fitness1m = normalize_fitness(max_fitness1)
norm_fitness2m = normalize_fitness(max_fitness2)
norm_fitness3m = normalize_fitness(max_fitness3)

# gen = [x for x in range(len(avg1))]
# print(gen)


plt.plot(gen, norm_fitness1, label='Line 1')
plt.plot(gen, norm_fitness2, label='Line 2')
plt.plot(gen, norm_fitness3, label='Line 3')
plt.xlabel("Generation")
plt.ylabel("Normalized Fitness")
plt.title("Normalized Average Fitness per Generation in All Lines")
plt.grid(True)
plt.legend()
output_chart_fp = os.getcwd() + "/GISFiles/output/avg fitness together.png"
plt.savefig(output_chart_fp)

plt.clf()
plt.plot(gen, norm_fitness1m, label='Line 1')
plt.plot(gen, norm_fitness2m, label='Line 2')
plt.plot(gen, norm_fitness3m, label='Line 3')
plt.xlabel("Generation")
plt.ylabel("Normalized Fitness")
plt.title("Normalized Max Fitness per Generation in All Lines")
plt.grid(True)
plt.legend()
output_chart_fp = os.getcwd() + "/GISFiles/output/max fitness together.png"
plt.savefig(output_chart_fp)