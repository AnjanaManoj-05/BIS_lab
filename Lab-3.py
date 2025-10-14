"""
pso_path_planner.py
Simple PSO path planner for a 2D environment with circular obstacles.
Particles = sequences of intermediate waypoints between start and goal.
Objective = path length + large collision penalty + smoothness term.

Requires: numpy, matplotlib
Run: python pso_path_planner.py
"""

import numpy as np
import matplotlib.pyplot as plt
import random

# ---------- Environment ----------
START = np.array([0.0, 0.0])
GOAL  = np.array([10.0, 10.0])
OBSTACLES = [  # (x, y, radius)
    (4.0, 5.0, 1.2),
    (7.0, 7.0, 1.0),
    (5.5, 2.7, 0.9),
]

X_LIMITS = (-1, 12)
Y_LIMITS = (-1, 12)

# ---------- PSO Hyperparameters ----------
NUM_PARTICLES = 40
NUM_WAYPOINTS = 6   # number of intermediate waypoints (excl. start/goal)
DIM = NUM_WAYPOINTS * 2
MAX_ITERS = 150

w = 0.7     # inertia
c1 = 1.4    # cognitive
c2 = 1.4    # social

VEL_CLAMP = 1.5

# Penalties / weights
COLLISION_PENALTY = 1e4
SMOOTHNESS_WEIGHT = 1.0


# ---------- Helper functions ----------
def path_points_from_particle(p):
    """Given particle vector p (length 2*num_waypoints), return full path as array of points including start and goal."""
    pts = p.reshape((NUM_WAYPOINTS, 2))
    full = np.vstack([START, pts, GOAL])
    return full

def path_length(path):
    return np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))

def collision_cost(path):
    cost = 0.0
    # sample along each segment to detect collisions (simple)
    samples_per_seg = 8
    for a, b in zip(path[:-1], path[1:]):
        for t in np.linspace(0, 1, samples_per_seg):
            p = a + t * (b - a)
            for (ox, oy, r) in OBSTACLES:
                if np.hypot(p[0]-ox, p[1]-oy) <= r:
                    cost += 1.0  # count one hit
    return cost * COLLISION_PENALTY

def smoothness_cost(path):
    # measure curvature-like cost: difference between consecutive segment directions
    segs = np.diff(path, axis=0)
    norms = np.linalg.norm(segs, axis=1, keepdims=True)
    norms[norms == 0] = 1e-6
    dirs = segs / norms
    diff = np.diff(dirs, axis=0)
    return SMOOTHNESS_WEIGHT * np.sum(np.linalg.norm(diff, axis=1))

def fitness(p):
    path = path_points_from_particle(p)
    return path_length(path) + smoothness_cost(path) + collision_cost(path)

def clip_particle(p):
    # clip waypoints to within environment bounds
    p = p.copy()
    for i in range(NUM_WAYPOINTS):
        x = p[2*i]
        y = p[2*i+1]
        x = np.clip(x, X_LIMITS[0], X_LIMITS[1])
        y = np.clip(y, Y_LIMITS[0], Y_LIMITS[1])
        p[2*i] = x
        p[2*i+1] = y
    return p

# ---------- Initialize swarm ----------
particles = [np.hstack([np.random.uniform(X_LIMITS[0], X_LIMITS[1], NUM_WAYPOINTS),
                        np.random.uniform(Y_LIMITS[0], Y_LIMITS[1], NUM_WAYPOINTS)]) for _ in range(NUM_PARTICLES)]
# but above interleaves all x then all y; we want x,y,x,y...
def random_particle():
    pts = []
    for _ in range(NUM_WAYPOINTS):
        pts.append(np.random.uniform(X_LIMITS[0], X_LIMITS[1]))
        pts.append(np.random.uniform(Y_LIMITS[0], Y_LIMITS[1]))
    return np.array(pts)

particles = [random_particle() for _ in range(NUM_PARTICLES)]
velocities = [np.random.uniform(-1,1,DIM) for _ in range(NUM_PARTICLES)]
pbest = [particles[i].copy() for i in range(NUM_PARTICLES)]
pbest_f = [fitness(pbest[i]) for i in range(NUM_PARTICLES)]
gbest_idx = int(np.argmin(pbest_f))
gbest = pbest[gbest_idx].copy()
gbest_f = pbest_f[gbest_idx]

# ---------- PSO main loop ----------
history = []
for it in range(MAX_ITERS):
    for i in range(NUM_PARTICLES):
        r1 = np.random.rand(DIM)
        r2 = np.random.rand(DIM)
        velocities[i] = (w * velocities[i]
                         + c1 * r1 * (pbest[i] - particles[i])
                         + c2 * r2 * (gbest - particles[i]))
        # clamp velocities
        velocities[i] = np.clip(velocities[i], -VEL_CLAMP, VEL_CLAMP)
        particles[i] = particles[i] + velocities[i]
        particles[i] = clip_particle(particles[i])

        f = fitness(particles[i])
        if f < pbest_f[i]:
            pbest[i] = particles[i].copy()
            pbest_f[i] = f

    # update global best
    idx = int(np.argmin(pbest_f))
    if pbest_f[idx] < gbest_f:
        gbest = pbest[idx].copy()
        gbest_f = pbest_f[idx]

    history.append(gbest_f)
    if (it+1) % 10 == 0 or it==0:
        print(f"Iter {it+1}/{MAX_ITERS}  Best Fitness: {gbest_f:.2f}")

# ---------- Results & plot ----------
best_path = path_points_from_particle(gbest)

fig, ax = plt.subplots(figsize=(7,7))
# plot obstacles
for (ox, oy, r) in OBSTACLES:
    circle = plt.Circle((ox,oy), r, color='r', alpha=0.4)
    ax.add_patch(circle)

# plot start & goal
ax.scatter(START[0], START[1], c='g', s=80, label='START')
ax.scatter(GOAL[0], GOAL[1], c='b', s=80, label='GOAL')

# plot best path
ax.plot(best_path[:,0], best_path[:,1], '-o', label='Best Path')
ax.set_xlim(X_LIMITS)
ax.set_ylim(Y_LIMITS)
ax.set_title("PSO Path Planning Result")
ax.legend()
plt.grid(True)
plt.show()

# plot fitness history
plt.figure()
plt.plot(history)
plt.xlabel("Iteration")
plt.ylabel("Best Fitness")
plt.title("PSO Convergence")
plt.grid(True)
plt.show()
