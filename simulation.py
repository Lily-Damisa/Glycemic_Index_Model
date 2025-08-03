import numpy as np
import matplotlib.pyplot as plt

# Parameters
population_size = 1000
generations = 50
antibiotic_strength = 0.5
mutation_rate = 0.1  # How much resistance can change during mutation

# Initialize population: random resistance values between 0 and 1
population = np.random.rand(population_size)

# Track average resistance over time
avg_resistance = []

for gen in range(generations):
    # Selection: survive if resistance >= antibiotic_strength
    survivors = population[population >= antibiotic_strength]
    
    if len(survivors) == 0:
        print(f"Population wiped out in generation {gen}")
        break

    # Reproduction: survivors reproduce to refill the population
    # Offspring inherit resistance + small mutation
    offspring = []
    while len(offspring) < population_size:
        parent = np.random.choice(survivors)
        mutation = np.random.uniform(-mutation_rate, mutation_rate)
        child = np.clip(parent + mutation, 0, 1)  # Keep resistance between 0 and 1
        offspring.append(child)
    
    population = np.array(offspring)
    avg_resistance.append(np.mean(population))

# Plot resistance over generations
plt.plot(avg_resistance)
plt.title("Average Bacterial Resistance Over Generations")
plt.xlabel("Generation")
plt.ylabel("Average Resistance")
plt.grid(True)
plt.show()
