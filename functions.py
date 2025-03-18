import numpy as np

# Функция Химмельблау
def himmelblau(x):
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

def himmelblau_gradient(x):
    df_dx = 4 * x[0] * (x[0] ** 2 + x[1] - 11) + 2 * (x[0] + x[1] ** 2 - 7)
    df_dy = 2 * (x[0] ** 2 + x[1] - 11) + 4 * x[1] * (x[0] + x[1] ** 2 - 7)
    return np.array([df_dx, df_dy])

# Функция Бута
def booth(x):
    return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2

def booth_gradient(x):
    df_dx = 2 * (x[0] + 2 * x[1] - 7) + 4 * (2 * x[0] + x[1] - 5)
    df_dy = 4 * (x[0] + 2 * x[1] - 7) + 2 * (2 * x[0] + x[1] - 5)
    return np.array([df_dx, df_dy])

# Функция Сферы
def sphere(x):
    return sum(xi**2 for xi in x)

def sphere_gradient(x):
    return 2 * np.array(x)

def rosenbrock(x, y):
    return (1 - x) ** 2 + 100 * (y - x * x) ** 2

def func1(x):
    """Основная функция"""
    return 2 * x[0]**2 + 2 * x[0] * x[1] + 2 * x[1]**2 - 4 * x[0] - 6 * x[1]

def dop(x):
    """Ограничение"""
    return x[0] + 2*x[1] - 2
