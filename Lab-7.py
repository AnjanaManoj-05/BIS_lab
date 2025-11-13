import numpy as np

def parallel_cellular_algorithm(f, grid_size=(10, 10), max_iterations=100, lb=-10, ub=10): rows, cols = grid_size
num_cells = rows * cols
# 1. Initialize grid with random values
X = np.random.uniform(lb, ub, size=(rows, cols))
# Evaluate fitness fitness = f(X)
# Track best global solution best_value = np.min(fitness)
best_pos = np.unravel_index(np.argmin(fitness), fitness.shape) best_solution = X[best_pos]
# Neighborhood offsets for 3x3 neighborhood neighbors_offset = [(-1,-1), (-1,0), (-1,1),
(0,-1), (0,0), (0,1),
(1,-1), (1,0), (1,1)]
for iteration in range(max_iterations): X_new = np.copy(X)
# 4. Update each cell in parallel logic for i in range(rows):
for j in range(cols):
# Collect neighbors neighbor_values = []
for dx, dy in neighbors_offset: ni, nj = i + dx, j + dy
if 0 <= ni < rows and 0 <= nj < cols: neighbor_values.append((X[ni, nj], fitness[ni, nj]))
# Get best neighbor (minimum fitness)
best_neighbor_value, _ = min(neighbor_values, key=lambda x: x[1])
# Update rule: average
X_new[i, j] = (X[i, j] + best_neighbor_value) / 2

# Boundary enforcement X_new = np.clip(X_new, lb, ub)

# Replace old values X = X_new

# Recalculate fitness fitness = f(X)

# Update global best current_best = np.min(fitness) if current_best < best_value:
best_value = current_best
best_pos = np.unravel_index(np.argmin(fitness), fitness.shape) best_solution = X[best_pos]
return best_solution, best_value

#
# Example: Optimize Sphere Function f(x) = x^2 #
def sphere_function(x):
return x**2 # works element-wise
best_x, best_val = parallel_cellular_algorithm( f=sphere_function,
grid_size=(10, 10), max_iterations=100, lb=-5,
ub=5
)
print("Best solution:", best_x) print("Best function value:", best_val)
