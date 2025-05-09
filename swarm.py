import numpy as np
from functions import rastrigin

# === Инициализация роя ===
def initialize_swarm(swarm_size, dimension, min_val, max_val):
    positions = np.random.rand(swarm_size, dimension) * (max_val - min_val) + min_val
    velocities = np.random.uniform(-1, 1, (swarm_size, dimension))
    best_local_positions = positions.copy()
    best_local_scores = np.array([rastrigin(pos) for pos in positions])
    return positions, velocities, best_local_positions, best_local_scores

# === Основной шаг PSO ===
def update_swarm(positions, velocities, best_local_positions, best_local_scores, global_best_position, inertia, alpha, beta, min_val, max_val):
    swarm_size, dimension = positions.shape
    r1 = np.random.rand(swarm_size, dimension)
    r2 = np.random.rand(swarm_size, dimension)

    cognitive = alpha * r1 * (best_local_positions - positions)
    social = beta * r2 * (global_best_position - positions)
    velocities = inertia * velocities + cognitive + social
    positions = positions + velocities

    # Clamp positions
    positions = np.clip(positions, min_val, max_val)

    scores = np.array([rastrigin(pos) for pos in positions])

    # Обновление локальных и глобальных лучших значений
    better_mask = scores < best_local_scores
    best_local_positions[better_mask] = positions[better_mask]
    best_local_scores[better_mask] = scores[better_mask]

    best_idx = np.argmin(best_local_scores)
    new_global_best_score = best_local_scores[best_idx]
    new_global_best_position = best_local_positions[best_idx]

    return positions, velocities, best_local_positions, best_local_scores, new_global_best_position, new_global_best_score
