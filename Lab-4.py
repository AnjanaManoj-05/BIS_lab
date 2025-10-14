"""
aco_tsp.py
Classic Ant Colony Optimization (ACO) implementation for TSP.
Plots the best tour found and convergence curve.

Requires: numpy, matplotlib
Run: python aco_tsp.py
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import random

# ---------- Problem definition ----------
NUM_CITIES = 20
np.random.seed(1)
cities = np.random.rand(NUM_CITIES, 2) * 100  # coordinates in 0..100

# Precompute distances
dist = np.zeros((NUM_CITIES, NUM_CITIES))
for i in range(NUM_CITIES):
    for j in range(NUM_CITIES):
        dist[i,j] = np.linalg.norm(cities[i]-cities[j])
# Avoid division by zero
heuristic = 1.0 / (dist + 1e-6)

# ---------- ACO hyperparameters ----------
NUM_ANTS = 40
NUM_ITERS = 200
alpha = 1.0   # pheromone importance
beta = 5.0    # heuristic importance
rho = 0.5     # evaporation rate
Q = 100.0     # pheromone deposit factor

# Initialize pheromone
tau0 = 1.0
tau = np.ones((NUM_CITIES, NUM_CITIES)) * tau0
best_tour = None
best_len = float('inf')
history = []

# ---------- Helper functions ----------
def tour_length(tour):
    total = sum(dist[tour[i], tour[i+1]] for i in range(len(tour)-1))
    return total

# ---------- ACO main loop ----------
for it in range(NUM_ITERS):
    all_tours = []
    all_lengths = []

    for k in range(NUM_ANTS):
        # build a tour using probabilistic selection
        start = random.randrange(NUM_CITIES)
        tour = [start]
        visited = set(tour)

        while len(tour) < NUM_CITIES:
            i = tour[-1]
            probs = []
            candidates = []
            for j in range(NUM_CITIES):
                if j not in visited:
                    val = (tau[i,j] ** alpha) * (heuristic[i,j] ** beta)
                    probs.append(val)
                    candidates.append(j)
            probs = np.array(probs)
            probs_sum = probs.sum()
            if probs_sum == 0:
                # pick random remaining
                nxt = random.choice([c for c in range(NUM_CITIES) if c not in visited])
            else:
                probs = probs / probs_sum
                nxt = np.random.choice(candidates, p=probs)
            tour.append(nxt)
            visited.add(nxt)

        tour.append(tour[0])  # return to start
        L = tour_length(tour)
        all_tours.append(tour)
        all_lengths.append(L)
        if L < best_len:
            best_len = L
            best_tour = tour.copy()

    # Evaporation
    tau *= (1 - rho)
    # Deposit (each ant)
    for tour, L in zip(all_tours, all_lengths):
        deposit = Q / L
        for i in range(len(tour)-1):
            a, b = tour[i], tour[i+1]
            tau[a,b] += deposit
            tau[b,a] += deposit  # symmetric

    history.append(best_len)
    if (it+1) % 10 == 0 or it==0:
        print(f"Iter {it+1}/{NUM_ITERS}  Best length: {best_len:.2f}")

# ---------- Plot best tour ----------
best_tour_coords = [cities[i] for i in best_tour]

plt.figure(figsize=(8,6))
xs = [p[0] for p in best_tour_coords]
ys = [p[1] for p in best_tour_coords]
plt.plot(xs, ys, '-o')
for idx, (x,y) in enumerate(cities):
    plt.text(x+0.8, y+0.8, str(idx), fontsize=8)
plt.title(f"Best tour length: {best_len:.2f}")
plt.grid(True)
plt.show()

# ---------- Plot convergence ----------
plt.figure()
plt.plot(history)
plt.xlabel("Iteration")
plt.ylabel("Best tour length")
plt.title("ACO Convergence")
plt.grid(True)
plt.show()
