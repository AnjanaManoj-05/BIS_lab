import random
import math

# Example coordinates for bus stops (x, y)
bus_stops = {
    'School': (50, 50),
    'Stop1': (10, 70),
    'Stop2': (20, 40),
    'Stop3': (30, 80),
    'Stop4': (60, 20),
    'Stop5': (80, 70)
}

# Parameters
POPULATION_SIZE = 6
GENERATIONS = 10
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.5

# Calculate Euclidean distance between two stops
def distance(stop1, stop2):
    x1, y1 = bus_stops[stop1]
    x2, y2 = bus_stops[stop2]
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

# Total distance of a route
def total_distance(route):
    dist = 0
    for i in range(len(route) - 1):
        dist += distance(route[i], route[i+1])
    dist += distance(route[-1], route[0])  # Return to school
    return dist

# Fitness function: inverse of distance
def fitness(route):
    return 1 / (total_distance(route) + 1e-6)  # small epsilon to avoid div by zero

# Create initial population
def create_population():
    stops = list(bus_stops.keys())
    stops.remove('School')
    population = []
    for _ in range(POPULATION_SIZE):
        route = stops[:]
        random.shuffle(route)
        route = ['School'] + route
        population.append(route)
    return population

# Selection (roulette wheel selection)
def select_parents(population):
    fitness_values = [fitness(route) for route in population]
    total_fitness = sum(fitness_values)
    probs = [f/total_fitness for f in fitness_values]
    parents = random.choices(population, probs, k=2)
    return parents

# Crossover (ordered crossover)
def crossover(parent1, parent2):
    if random.random() > CROSSOVER_RATE:
        return parent1[:], parent2[:]

    start, end = sorted(random.sample(range(1, len(parent1)), 2))
    child1_middle = parent1[start:end]
    child2_middle = parent2[start:end]

    def fill_child(parent, middle):
        child = ['School'] + [stop for stop in parent[1:] if stop not in middle]
        child[start:start] = middle
        return child

    return fill_child(parent2, child1_middle), fill_child(parent1, child2_middle)

# Mutation (swap two stops)
def mutate(route):
    if random.random() < MUTATION_RATE:
        i, j = random.sample(range(1, len(route)), 2)
        route[i], route[j] = route[j], route[i]
    return route

# Genetic Algorithm main loop
def genetic_algorithm():
    population = create_population()
    best_route = None
    best_dist = float('inf')

    for gen in range(1, GENERATIONS+1):
        new_population = []
        print(f"Generation {gen}:")
        for _ in range(POPULATION_SIZE // 2):
            parent1, parent2 = select_parents(population)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1)
            child2 = mutate(child2)
            new_population.extend([child1, child2])
        
        population = new_population

        # Track best route
        for route in population:
            dist = total_distance(route)
            if dist < best_dist:
                best_dist = dist
                best_route = route[:]

        # Print each generation's best
        print(f"  Best distance: {best_dist:.2f}, Route: {best_route}")

    print("\nOptimal bus route found:")
    print(f"Distance: {best_dist:.2f}")
    print(f"Route: {best_route}")

# Run the GA
genetic_algorithm()
