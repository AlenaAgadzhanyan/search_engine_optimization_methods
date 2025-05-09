import tkinter as tk
import numpy as np
import random
from functions import *

def initialize_bees(num_bees, fitness_func, bounds):
    bees = []
    for _ in range(num_bees):
        coords = np.array([random.uniform(b[0], b[1]) for b in bounds])
        fitness = fitness_func(coords)
        bees.append((coords, fitness))
    return bees

def explore_bee(current_site, best_sites, elite_radius, perspective_radius, fitness_func):
        is_elite = any(np.array_equal(current_site[0], site[0]) for site in best_sites[:len(best_sites)//2])
        
        radius = elite_radius if is_elite else perspective_radius
        
        x = current_site[0][0] + random.uniform(-radius, radius)
        y = current_site[0][1] + random.uniform(-radius, radius)
        return ((x, y), fitness_func(np.array([x, y])))
