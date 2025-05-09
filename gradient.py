import numpy as np
from functions import *

def gradient_descent(x_start, y_start, step_size, max_iterations, selected_function):
    epsilon1 = 0.001  # Критерий окончания |f(xk)| < ε1
    epsilon2 = 0.001  # Критерий остановки ||xk+1 - xk|| < ε2 и ||f(xk+1) - f(xk)|| < ε2
    epsilon = 0.001

    x = np.array([x_start, y_start])
    history = []
    k = 0

    while True:
        if selected_function == "Функция Химмельблау":
            f_xk = himmelblau(x)
            grad = himmelblau_gradient(x)
        elif selected_function == "Функция Бута":
            f_xk = booth(x)
            grad = booth_gradient(x)
        elif selected_function == "Функция Сферы":
            f_xk = sphere(x)
            grad = sphere_gradient(x)

        if abs(f_xk) < epsilon1:
            history.append((x[0], x[1], f_xk))
            break

        if k >= max_iterations:
            history.append((x[0], x[1], f_xk))
            break

        tk = step_size

        while True:
            x_new = x - tk * grad

            if selected_function == "Функция Химмельблау":
                f_xk_new = himmelblau(x_new)
            elif selected_function == "Функция Бута":
                f_xk_new = booth(x_new)
            elif selected_function == "Функция Сферы":
                f_xk_new = sphere(x_new)

            if f_xk_new - f_xk < 0 or np.abs(f_xk_new - f_xk) < epsilon * (np.linalg.norm(f_xk) ** 2):
                break
            else:
                tk /= 2

        if np.linalg.norm(x_new - x) < epsilon2 and np.linalg.norm(f_xk_new - f_xk) < epsilon2:
            history.append((x_new[0], x_new[1], f_xk_new))
            break
        else:
            x = x_new
            k += 1
            history.append((x[0], x[1], f_xk_new))

    return history
