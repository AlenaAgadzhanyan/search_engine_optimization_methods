# import numpy as np

# def bacteria_foraging_optimization(n, function, iteration, num_swimming_steps, step_size, dispersion_probability, Nc=2, lb=-5, ub=5):
#     dispersion_probability /= 100
#     agents = np.random.uniform(lb, ub, (n, 2))
#     all_positions = [agents.copy()]
#     fitness_history = []

#     C_list = [step_size - step_size * 0.9 * t / iteration for t in range(iteration)]

#     # Инициализация глобального минимума
#     gbest_position = agents[0].copy()
#     gbest_value = function(gbest_position)

#     for t in range(iteration):
#         step_size = C_list[t]
#         J_chem = []

#         for _ in range(Nc):  # Nc хемотактических шагов
#             for i in range(n):
#                 direction = np.random.uniform(-1, 1, 2)
#                 direction /= np.linalg.norm(direction)  # Нормализуем направление

#                 current_pos = agents[i].copy()
#                 current_val = function(current_pos)

#                 # Первый шаг в направлении
#                 new_pos = current_pos + step_size * direction
#                 new_val = function(new_pos)

#                 if new_val < current_val:
#                     agents[i] = new_pos
#                     current_val = new_val

#                     # Плавание Ns шагов
#                     for _ in range(1, num_swimming_steps):
#                         next_pos = agents[i] + step_size * direction
#                         next_val = function(next_pos)

#                         if next_val < current_val:
#                             agents[i] = next_pos
#                             current_val = next_val
#                         else:
#                             break

#                 # Обновление глобального минимума, если найдено лучше
#                 if current_val < gbest_value:
#                     gbest_value = current_val
#                     gbest_position = agents[i].copy()

#             # Запоминаем значения приспособленности после каждой хемотактической итерации
#             J_chem.append(np.array([function(agent) for agent in agents]))

#         # Добавляем хемотаксическую историю в общую историю
#         fitness_history.append(J_chem[-1])

#         # Репродукция: оценка здоровья каждого агента
#         health = [(sum(fitness_history[k][i] for k in range(len(fitness_history))), i) for i in range(n)]
#         health.sort()
#         half = n // 2
#         survivors = [agents[i].copy() for (_, i) in health[:half]]
#         agents = np.array(survivors * 2)  # Дублируем выживших

#         # Элиминация-дисперсия: с вероятностью Ped заменяем агента на случайного
#         for i in range(n):
#             if np.random.rand() < dispersion_probability:
#                 agents[i] = np.random.uniform(lb, ub, 2)

#         # Гарантируем, что лучший агент сохраняется
#         agents[0] = gbest_position.copy()

#         all_positions.append(agents.copy())


#     return all_positions


import numpy as np

def bacteria_foraging_optimization(n, function, iteration, num_swimming_steps, step_size, dispersion_probability, Nc=2, lb=-5, ub=5):
    dispersion_probability /= 100  # Преобразуем проценты в дробную величину
    agents = np.random.uniform(lb, ub, (n, 2))  # Инициализация агентов случайно
    all_positions = [agents.copy()]  # Список для хранения всех позиций агентов
    fitness_history = []  # История значений фитнес-функции

    C_list = [step_size - step_size * 0.9 * t / iteration for t in range(iteration)]  # Список для шага

    # Инициализация глобального максимума
    gbest_position = agents[0].copy()
    gbest_value = function(gbest_position)

    for t in range(iteration):
        step_size = C_list[t]
        J_chem = []  # История фитнес-функций на данном шаге

        for _ in range(Nc):  # Nc хемотактических шагов
            for i in range(n):
                direction = np.random.uniform(-1, 1, 2)  # Случайное направление
                direction /= np.linalg.norm(direction)  # Нормализуем направление

                current_pos = agents[i].copy()
                current_val = function(current_pos)

                # Первый шаг в направлении
                new_pos = current_pos + step_size * direction
                new_val = function(new_pos)

                if new_val > current_val:  # Условие для максимизации
                    agents[i] = new_pos
                    current_val = new_val

                    # Плавание Ns шагов
                    for _ in range(1, num_swimming_steps):
                        next_pos = agents[i] + step_size * direction
                        next_val = function(next_pos)

                        if next_val > current_val:  # Условие для максимизации
                            agents[i] = next_pos
                            current_val = next_val
                        else:
                            break

                # Обновление глобального максимума, если найдено лучшее значение
                if current_val > gbest_value:
                    gbest_value = current_val
                    gbest_position = agents[i].copy()

            # Запоминаем значения фитнес-функции после каждого хемотактического шага
            J_chem.append(np.array([function(agent) for agent in agents]))

        # Добавляем хемотаксическую историю в общую историю
        fitness_history.append(J_chem[-1])

        # Репродукция: оценка здоровья каждого агента
        health = [(sum(fitness_history[k][i] for k in range(len(fitness_history))), i) for i in range(n)]
        health.sort(reverse=True)  # Для максимизации отсортируем в обратном порядке
        half = n // 2
        survivors = [agents[i].copy() for (_, i) in health[:half]]  # Выбираем лучших агентов
        agents = np.array(survivors * 2)  # Дублируем выживших

        # Элиминация-дисперсия: с вероятностью Ped заменяем агента на случайного
        for i in range(n):
            if np.random.rand() < dispersion_probability:
                agents[i] = np.random.uniform(lb, ub, 2)

        # Гарантируем, что лучший агент сохраняется
        agents[0] = gbest_position.copy()

        all_positions.append(agents.copy())  # Добавляем текущие позиции агентов в историю

    return all_positions
