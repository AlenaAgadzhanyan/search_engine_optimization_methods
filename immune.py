import random
from operator import itemgetter

def init_population(func, agents_numb, bounds):
    population = [[random.uniform(bounds[0][0], bounds[0][1]),
                   random.uniform(bounds[1][0], bounds[1][1])] for _ in range(agents_numb)]
    evaluated = [[x, y, func([x, y])] for x, y in population]
    return evaluated

def immune_step(agents, func, best_numb, clon_numb, best_clon_numb, coef, bounds, rev=False):
    best = sorted(agents, key=itemgetter(2), reverse=rev)[:best_numb]
    clones = []

    for agent in best:
        for _ in range(clon_numb):
            x = agent[0] + coef * random.uniform(-1, 1)
            y = agent[1] + coef * random.uniform(-1, 1)
            
            # Ограничиваем значения в пределах границ
            x = max(bounds[0][0], min(bounds[0][1], x))
            y = max(bounds[1][0], min(bounds[1][1], y))
            
            z = func([x, y])
            clones.append([x, y, z])

    top_clones = sorted(clones, key=itemgetter(2), reverse=rev)[:best_clon_numb]
    new_population = agents + top_clones
    new_population = sorted(new_population, key=itemgetter(2), reverse=rev)[:len(agents)]

    return new_population

def get_best(agents):
    return min(agents, key=itemgetter(2))