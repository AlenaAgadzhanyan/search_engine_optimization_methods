import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
import numpy as np


def GeneticAlgorithm(frame, root, ax, canvas):

    # Функция Розенброка для оптимизации
    def rosenbrock_function(x, y):
        return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

    # Оператор селекции (выбор лучших особей)
    def selection(population, fitness_scores):
        # Выбираем двух наилучших особей
        best_indices = np.argsort(fitness_scores)[:2]
        return [population[i] for i in best_indices]

    # Оператор кроссовера (одноточечный кроссовер)
    def crossover(parent1, parent2):
        crossover_point = np.random.randint(1, len(parent1))
        child = np.hstack((parent1[:crossover_point], parent2[crossover_point:]))
        return child

    # Оператор мутации
    def mutate(individual, mutation_rate):
        mutation_indices = np.random.rand(len(individual)) < mutation_rate
        individual[mutation_indices] += np.random.uniform(-0.5, 0.5)
        return individual


    def run_optimization():
        # Генерация сетки для графика целевой функции
        x_range = np.linspace(-5, 5, 100)
        y_range = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(x_range, y_range)
        Z = rosenbrock_function(X, Y)

        population_size = int(x_var.get())
        num_generations = int(y_var.get())
        # рандомно задаем популяцию от -5 до 5
        population = np.random.uniform(low=-5, high=5, size=(population_size, 2))

        # для записи результатов
        results = []
        results_text.config(state=tk.NORMAL)
        results_text.delete(1.0, tk.END)
        for generation in range(num_generations):
            # Расчет значений функции для текущей популяции
            fitness_scores = np.array([rosenbrock_function(x, y) for x, y in population])

            # Выбор лучших особей
            selected_individuals = selection(population, fitness_scores)

            # Оператор кроссовера и мутации
            children = []
            for i in range(0, population_size, 2):
                child1 = crossover(selected_individuals[0], selected_individuals[1])
                child2 = crossover(selected_individuals[1], selected_individuals[0])
                child1 = mutate(child1, mutation_rate=0.1)  # Пример вероятности мутации
                child2 = mutate(child2, mutation_rate=0.1)
                children.extend([child1, child2])

            ax.cla()
            # Построение поверхности графика целевой функции
            ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title("Генетический алгоритм")

            for i in range(len(fitness_scores)):
                best_individual = population[i]
                ax.scatter(best_individual[0], best_individual[1], fitness_scores[i], color='red',
                           s=10)

            # Обновление популяции
            population = np.array(children)

            # Нахождение лучшей особи на текущей итерации
            best_fitness = np.min(fitness_scores)
            best_individual = population[np.argmin(fitness_scores)]

            # Вывод лучшего решения на текущей итерации
            print(f"Поколение {generation}: Лучшее решение - {best_individual}, Значение функции - {best_fitness}")

            results.append((best_individual[0], best_individual[1], generation, best_fitness))
            results_text.insert(tk.END,
                                f"Поколение {generation}: Лучшее решение ({best_individual[0]:.2f}, {best_individual[1]:.2f}), Значение функции: {best_fitness:.7f}\n")
            results_text.yview_moveto(1)
            canvas.draw()
            root.update()

        # Нахождение лучшего решения после всех итераций
        final_fitness_scores = np.array([rosenbrock_function(x, y) for x, y in population])
        best_index = np.argmin(final_fitness_scores)
        best_solution = population[best_index]
        best_fitness_value = final_fitness_scores[best_index]

        results_text.insert(tk.END,
                            f"\nОптимизация завершена. Лучшее решение - {best_solution}, Значение функции - {best_fitness_value}")
        results_text.yview_moveto(1)
        ax.scatter(best_solution[0], best_solution[1], best_fitness_value, color='black', marker='x', s=60)
        results_text.config(state=tk.DISABLED)


    def clear_results():
        results_text.config(state=tk.NORMAL)
        results_text.delete("1.0", tk.END)
        results_text.config(state=tk.DISABLED)

   

    # --- Colors ---
    bg_color = "#3d6466"
    fg_color = "black"    
    button_color = "#28393a"
    font_large = ("Arial", 14, "bold")
    font_small = ("Arial", 10)

    # --- Styles ---
    style = ttk.Style()
    style.configure("TLabel", background=bg_color, foreground=fg_color, font=font_small)
    style.configure("TButton", background=button_color, foreground=fg_color, font=font_small)
    style.configure("TEntry", fieldbackground=bg_color, foreground=fg_color, font=font_small)
    style.configure("TCombobox", fieldbackground=bg_color, foreground=fg_color, font=font_small)
    style.configure("Horizontal.TSeparator", background=fg_color) # Separator color

    param_frame2 = frame

    # Background color for the frame
    param_frame2.configure(bg=bg_color)

    # --- Labels ---
    ttk.Label(param_frame2, text="Параметры гентического алгоритма", font=font_large).grid(row=0, column=0, pady=5, columnspan=2) # Убрал интервал
    ttk.Label(param_frame2, text="Pазмер популяции:", font=font_small).grid(row=1, column=0, sticky="w", padx=10)
    ttk.Label(param_frame2, text="Итерации:", font=font_small).grid(row=2, column=0, sticky="w", padx=10)

    # --- Entry Fields ---
    x_var = tk.DoubleVar(value=100)
    y_var = tk.DoubleVar(value=100)
    x_entry = ttk.Entry(param_frame2, textvariable=x_var, width=15)
    y_entry = ttk.Entry(param_frame2, textvariable=y_var, width=15)

    x_entry.grid(row=1, column=1, padx=10, pady=5, sticky="e")
    y_entry.grid(row=2, column=1, padx=10, pady=5, sticky="e")

    # --- Separator 1 ---
    separator1 = ttk.Separator(param_frame2, orient="horizontal")
    separator1.grid(row=3, column=0, columnspan=2, sticky="ew", pady=5, padx=10) # Убрал интервал

    # --- Function Parameters ---
    ttk.Label(param_frame2, text="Функция и отображение ее графика", font=font_large).grid(row=4, column=0, pady=5, columnspan=2) # Убрал интервал
    ttk.Label(param_frame2, text="Выберите функцию:", font=font_small).grid(row=5, column=0, sticky="w", padx=10)
    function_choices = ["Функция Розенброка"]
    function_var = tk.StringVar(value=function_choices[0])
    function_menu = ttk.Combobox(param_frame2, textvariable=function_var, values=function_choices, width=22)
    function_menu.grid(row=5, column=1, pady=5, padx=10, sticky="e")

    # --- Buttons Frame ---
    button_frame = tk.Frame(param_frame2, bg=bg_color)
    button_frame.grid(row=6, column=0, columnspan=2, pady=5) # Убрал интервал

    # --- Run Button ---
    apply_settings_button = ttk.Button(button_frame, text="Запуск", command=run_optimization, style="My.TButton")
    apply_settings_button.pack(side=tk.LEFT, padx=10)

    # --- Clear Button ---
    clear_button = ttk.Button(button_frame, text="Очистить", command=clear_results, style="My.TButton")
    clear_button.pack(side=tk.LEFT, padx=10)

    # --- Results Display ---
    ttk.Label(param_frame2, text="Результаты оптимизации", font=font_large).grid(row=7, column=0, pady=5, columnspan=2) # Убрала интервалы
    results_text = scrolledtext.ScrolledText(param_frame2, wrap=tk.WORD, height=10, width=40, padx=2, state=tk.DISABLED, bg="#4a8789", fg=fg_color)
    results_text.grid(row=8, column=0, padx=10, columnspan=2, sticky="ew")