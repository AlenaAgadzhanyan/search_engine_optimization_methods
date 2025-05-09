import numpy as np
from immune import immune_step

def hybrid_algorithm(
    func, n, iterations, swim_steps, step_size, dispersion_prob,
    best_numb, clon_numb, best_clon_numb, bounds, maximize=False, Nc=2
):
    lb, ub = bounds[0][0], bounds[0][1]
    dispersion_prob /= 100
    agents = np.random.uniform(lb, ub, (n, 2))
    all_positions = [agents.copy()]
    fitness_history = []

    C_list = [step_size - step_size * 0.9 * t / iterations for t in range(iterations)]
    gbest_position = agents[0].copy()
    gbest_value = func(gbest_position)

    def is_better(a, b):
        return a > b if maximize else a < b

    for t in range(iterations):
        step_size = C_list[t]
        J_chem = []

        for _ in range(Nc):
            for i in range(n):
                direction = np.random.uniform(-1, 1, 2)
                direction /= np.linalg.norm(direction)

                current_pos = agents[i].copy()
                current_val = func(current_pos)

                new_pos = current_pos + step_size * direction
                new_val = func(new_pos)

                if is_better(new_val, current_val):
                    agents[i] = new_pos
                    current_val = new_val

                    for _ in range(1, swim_steps):
                        next_pos = agents[i] + step_size * direction
                        next_val = func(next_pos)

                        if is_better(next_val, current_val):
                            agents[i] = next_pos
                            current_val = next_val
                        else:
                            break

                if is_better(current_val, gbest_value):
                    gbest_value = current_val
                    gbest_position = agents[i].copy()

            J_chem.append(np.array([func(agent) for agent in agents]))

        fitness_history.append(J_chem[-1])

        # Иммунная фаза
        evaluated = [[agent[0], agent[1], func(agent)] for agent in agents]
        evaluated = immune_step(
            evaluated, func, best_numb, clon_numb, best_clon_numb,
            coef=step_size, bounds=[(lb, ub), (lb, ub)], rev=maximize
        )
        agents = np.array([[x, y] for x, y, _ in evaluated])

        # Репродукция
        health = [(sum(fitness_history[k][i] for k in range(len(fitness_history))), i) for i in range(n)]
        health.sort(reverse=maximize)
        half = n // 2
        survivors = [agents[i].copy() for (_, i) in health[:half]]
        agents = np.array(survivors * 2)

        for i in range(n):
            if np.random.rand() < dispersion_prob:
                agents[i] = np.random.uniform(lb, ub, 2)

        agents[0] = gbest_position.copy()
        all_positions.append(agents.copy())

    return all_positions, gbest_position
