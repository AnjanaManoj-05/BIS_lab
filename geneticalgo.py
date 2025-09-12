import random

# -----------------------------
# 1. Define the Problem
# -----------------------------
# Example: Sphere function (minimization)
def objective_function(x):
    return sum([xi**2 for xi in x])  # Minimum is 0 at x = [0, 0, ..., 0]

# -----------------------------
# Gene Expression Algorithm (GEA)
# -----------------------------
class GEA:
    def __init__(self, func, num_genes, pop_size=20, mutation_rate=0.1,
                 crossover_rate=0.8, max_generations=100, gene_range=(-10,10),
                 maximize=False):
        # 2. Initialize Parameters
        self.func = func
        self.num_genes = num_genes
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_generations = max_generations
        self.gene_range = gene_range
        self.maximize = maximize

        # 3. Initialize Population
        self.population = self.initialize_population()
        self.best_solution = None
        self.best_fitness = float('-inf') if maximize else float('inf')

    # Step 3: Initialize Population
    def initialize_population(self):
        return [[random.uniform(*self.gene_range) for _ in range(self.num_genes)]
                for _ in range(self.pop_size)]

    # Step 4: Evaluate Fitness
    def evaluate_fitness(self, population):
        fitness_values = [self.func(ind) for ind in population]
        return fitness_values

    # Step 5: Selection (Tournament Selection)
    def selection(self, population, fitness_values, k=3):
        selected = []
        for _ in range(self.pop_size):
            tournament = random.sample(list(zip(population, fitness_values)), k)
            if self.maximize:
                winner = max(tournament, key=lambda t: t[1])[0]
            else:
                winner = min(tournament, key=lambda t: t[1])[0]
            selected.append(winner)
        return selected

    # Step 6: Crossover (Single-point)
    def crossover(self, parent1, parent2):
        point = random.randint(1, self.num_genes-1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2

    # Step 7: Mutation (Gaussian mutation)
    def mutate(self, individual):
        for i in range(self.num_genes):
            if random.random() < self.mutation_rate:
                individual[i] += random.gauss(0, 1)  # Small random change
                # Keep within gene range
                individual[i] = max(min(individual[i], self.gene_range[1]), self.gene_range[0])

    # Step 8: Gene Expression (Tracking best solution)
    def track_best(self, population, fitness_values):
        for ind, fit in zip(population, fitness_values):
            if (self.maximize and fit > self.best_fitness) or (not self.maximize and fit < self.best_fitness):
                self.best_solution = ind.copy()
                self.best_fitness = fit

    # Step 9: Iterate with printing each generation
    def run(self):
        for generation in range(1, self.max_generations + 1):
            # Evaluate current population
            fitness_values = self.evaluate_fitness(self.population)
            self.track_best(self.population, fitness_values)

            # Print best of this generation
            print(f"Generation {generation}: Best fitness = {self.best_fitness:.6f}, Best solution = {self.best_solution}")

            # Selection
            parents = self.selection(self.population, fitness_values)

            # Crossover and Mutation to produce offspring
            offspring = []
            while len(offspring) < self.pop_size:
                parent1, parent2 = random.sample(parents, 2)
                if random.random() < self.crossover_rate:
                    child1, child2 = self.crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                self.mutate(child1)
                self.mutate(child2)

                offspring.append(child1)
                if len(offspring) < self.pop_size:
                    offspring.append(child2)

            # Replace population with offspring
            self.population = offspring

        # Step 10: Output Best Solution
        return self.best_solution, self.best_fitness

# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    num_genes = 5  # Number of variables in the optimization problem
    gea = GEA(func=objective_function, num_genes=num_genes,
              pop_size=30, mutation_rate=0.1, crossover_rate=0.8,
              max_generations=20, gene_range=(-5,5), maximize=False)
    
    best_solution, best_fitness = gea.run()
    print("\nFinal Best Solution:")
    print("Best solution found:", best_solution)
    print("Best fitness:", best_fitness)
