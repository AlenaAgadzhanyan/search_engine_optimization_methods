import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
import numpy as np
import random



# Класс для представления пчелы
class Bee:
    """
     coords - координаты поля для пчелы
     fitness - значение фитнесс-функции для центра данного участка
    """
    def __init__(self, coords, fitness):
        self.coords = coords
        self.fitness = fitness


class BeeAlgorithm:
    def __init__(self, num_scouts, elite_radius, perspective_radius, num_elite, num_perspective, agents_per_perspective,
                 agents_per_elite, bounds, max_epochs, stagnation_limit, fitness_function):
        '''
        - num_scouts (int): Количество пчел-разведчиков в популяции.
        - elite_radius (float): Радиус элитных участков для каждой пчелы.
        - perspective_radius (float): Радиус перспективных участков для каждой пчелы.
        - num_elite (int): Количество элитных участков для обновления координат пчелы.
        - num_perspective (int): Количество перспективных участков для обновления координат пчелы.
        - agents_per_perspective (int): Количество агентов, отправляемых на каждый перспективный участок.
        - agents_per_elite (int): Количество агентов, отправляемых на каждый элитный участок.
        - bounds (listусловие останова).
        - fitness_function: заданная фитнесс-функция
        '''
        self.num_scouts = num_scouts
        self.elite_radius = elite_radius
        self.perspective_radius = perspective_radius
        self.num_elite = num_elite
        self.num_perspective = num_perspective
        self.agents_per_perspective = agents_per_perspective
        self.agents_per_elite = agents_per_elite
        self.bounds = bounds
        self.max_epochs = max_epochs
        self.stagnation_limit = stagnation_limit
        self.best_bees = []
        self.fitness_function = fitness_function

    def set_options(self, root, ax, canvas, results_text,bound_start,bound_end,target_func):
        self.canvas = canvas
        self.root = root
        self.ax = ax
        self.results_text = results_text
        self.ax = ax
        self.bound_start = bound_start
        self.bound_end = bound_end
        self.target_func = target_func

    def initialize_bees(self):
        bees = []
        for _ in range(self.num_scouts):
            # Инициализация случайных координат для каждой пчелы в пределах заданных границ
            coords = np.array([random.uniform(self.bounds[i][0], self.bounds[i][1]) for i in range(len(self.bounds))], dtype='float')
            bees.append(Bee(coords, self.fitness_function(coords)))
        return bees

    def optimize(self):
        # инициализация начальной популяции пчел
        bees = self.initialize_bees()

        stagnation_count = 0
        best_fitness = float('inf')

        for epoch in range(self.max_epochs):
            # Сортировка пчел по их приспособленности (лучшие впереди)
            bees = sorted(bees, key=lambda bee: bee.fitness)
            self.best_bees = bees[:self.num_elite]

            x_range = np.linspace(self.bound_start, self.bound_end, 100)
            y_range = np.linspace(self.bound_start, self.bound_end, 100)
            X, Y = np.meshgrid(x_range, y_range)
            Z = np.zeros_like(X)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    Z[i, j] = self.target_func(np.array([X[i, j], Y[i, j]]))

            self.ax.cla()
            self.canvas.draw()
            self.ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_zlabel('Z')
            self.ax.set_xticks(np.arange(self.bound_start, self.bound_end + 1, 2))
            self.ax.set_yticks(np.arange(self.bound_start, self.bound_end + 1, 2))

            for i in range(self.num_scouts):
                # Исследование окружения для каждой пчелы-разведчика
                self.explore(bees[i])
                # print(bees[i].fitness,bees[i].coords[0],bees[i].coords[1])
                self.ax.scatter(bees[i].coords[0], bees[i].coords[1], bees[i].fitness, color='black',
                           s=10)

            # Выбор лучших пчел из текущей эпохи
            bees = self.select_best(bees)

            # print(bees[i].fitness)
            # Проверка условия стагнации
            current_best_fitness = self.best_bees[0].fitness
            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                stagnation_count = 0
            else:
                stagnation_count += 1

            if stagnation_count >= self.stagnation_limit:
                self.results_text.insert(tk.END,f"\nСтагнация. Оптимизация остановлена на итерации {epoch}.\n")
                break


            # self.ax.scatter(self.best_bees[0].coords[0], self.best_bees[0].coords[1], self.best_bees[0].fitness, c="black")
            self.results_text.insert(tk.END,
                                f"Итерация {epoch}: Лучшее решение ({self.best_bees[0].coords[0]:.8f}, {self.best_bees[0].coords[1]:.8f}, {self.best_bees[0].fitness:.8f})\n")
            self.canvas.draw()
            self.results_text.yview_moveto(1)
            self.root.update()
            # print(f'Лучшая пчела в {self.best_bees[0].coords} и фитнесс {self.best_bees[0].fitness}')

        # Последняя сортировка и выбор лучших пчел
        self.best_bees = sorted(bees, key=lambda bee: bee.fitness)[:self.num_elite]
        return self.best_bees[0]

    def explore(self, bee):
        # Случайная фаза исследования соседнего пространства
        phi = random.uniform(-1, 1)
        phi_elite = random.uniform(0, self.elite_radius)
        phi_perspective = random.uniform(0, self.perspective_radius)

        # Выбор случайного направления для каждой координаты
        directions = [random.uniform(-1, 1) for _ in range(len(bee.coords))]

        # Новые координаты для текущей пчелы
        new_coords = [bee.coords[i] + phi * (bee.coords[i] - self.best_bees[0].coords[i]) +
                      phi_elite * (bee.coords[i] - random.choice(self.best_bees).coords[i]) +
                      phi_perspective * random.choice(self.best_bees).coords[i] *
                      directions[i] for i in range(len(bee.coords))]

        # Ограничение координат в пределах заданных границ
        new_coords = np.array(
            [max(min(new_coords[i], self.bounds[i][1]), self.bounds[i][0]) for i in range(len(new_coords))])

        # Вычисление новой приспособленности для пчелы
        new_fitness = self.fitness_function(new_coords)

        # Обновление координат и приспособленности, если новая точка лучше
        if new_fitness < bee.fitness:
            bee.coords = new_coords
            bee.fitness = new_fitness

    def select_best(self, bees):
        # Сортировка всех пчел
        bees.sort(key=lambda bee: bee.fitness)
        # Выбор лучших мест и инициализация новых пчел для заполнения оставшихся мест
        return bees[:self.num_perspective] + self.initialize_bees()[:self.num_scouts - self.num_perspective]




def BeesAlgorithm(frame,root,ax,canvas):

        # Функция Розенброка для оптимизации
        def himel_function(x_arr):
            x, y = x_arr[0], x_arr[1]
            return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2

        def rosenbrock_function(x_arr):
            x, y = x_arr[0], x_arr[1]
            return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

        def rastrigin(x_arr):
            size = len(x_arr)
            return 10 * size + np.sum(x_arr ** 2 - 10 * np.cos(2 * np.pi * x_arr))

        def run_optimization():
            ax.cla()
            function_choice = function_var.get()
            target_func = himel_function
            if function_choice == "Функция Химмельблау":
                target_func = himel_function
            elif function_choice == "Функция Розенброка":
                target_func = rosenbrock_function
            elif function_choice == "Функция Растригина":
                target_func = rastrigin



            iterations=int(iteration.get())
            scout = int(scouts.get())  # разведчики
            perspective_B = int(perspective_b.get())
            best_B = int(best_b.get())  # лучшие пчелы
            perspective_A = int(perspective_a.get())
            best_A  = int(best_a.get())
            size_A = int(size_a.get())
            stop_entry = int(stop.get())

            bound_start = float(-4)
            bound_end = float(4)
            bounds = [(bound_start, bound_end) for i in range(2)]


            x_range = np.linspace(bound_start, bound_end, 100)
            y_range = np.linspace(bound_start, bound_end, 100)
            X, Y = np.meshgrid(x_range, y_range)
            Z = np.zeros_like(X)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    Z[i, j] = target_func(np.array([X[i, j], Y[i, j]]))


            results_text.config(state=tk.NORMAL)
            results_text.delete(1.0, tk.END)
            algorithm = BeeAlgorithm(scout, size_A, size_A, best_A, perspective_A,
                                     perspective_B, best_B, bounds, iterations, stop_entry,
                                     target_func)
            algorithm.set_options(root, ax, canvas, results_text,bound_start,bound_end,target_func)
            best_bee = algorithm.optimize()
            ax.scatter(best_bee.coords[0], best_bee.coords[1], best_bee.fitness, c="red")
            results_text.insert(tk.END,
                                f"Лучшее решение ({best_bee.coords[0]:.8f}, {best_bee.coords[1]:.8f}, {best_bee.fitness:.8f})\n")

            canvas.draw()
            root.update()


        param_frame2 = frame
        bg_color = "#3d6466"
        param_frame2.configure(bg=bg_color)

        # Параметры задачи
        ttk.Label(param_frame2, text="Инициализация значений", font=("Helvetica", 12,"bold")).grid(row=0, column=0, pady=15)
        ttk.Label(param_frame2, text="Итераций", font=("Helvetica", 10)).grid(row=1, column=0)
        ttk.Label(param_frame2, text="Разведчики", font=("Helvetica", 10)).grid(row=2, column=0)
        ttk.Label(param_frame2, text="Пчел в перспективном участке", font=("Helvetica", 10)).grid(row=3, column=0)
        ttk.Label(param_frame2, text="Пчел в лучшем участке", font=("Helvetica", 10)).grid(row=4, column=0)
        ttk.Label(param_frame2, text="Перспективных участков", font=("Helvetica", 10)).grid(row=5, column=0)
        ttk.Label(param_frame2, text="Лучших участков", font=("Helvetica", 10)).grid(row=6, column=0)
        ttk.Label(param_frame2, text="Размер участков", font=("Helvetica", 10)).grid(row=7, column=0)
        ttk.Label(param_frame2, text="Критерий останова", font=("Helvetica", 10)).grid(row=8, column=0)


        iteration = tk.IntVar(value=200)
        scouts = tk.IntVar(value=20) #разведчики
        perspective_b = tk.IntVar(value=10) #перспективных пчел
        best_b = tk.IntVar(value=20) #лучшие пчелы
        perspective_a  = tk.IntVar(value=3) #перпективных участков
        best_a = tk.IntVar(value=1)  # лучших участков
        size_a = tk.DoubleVar(value=0.5)  # размер участков
        stop = tk.DoubleVar(value=20)  # задержка

        iteration_entry = ttk.Entry(param_frame2, textvariable=iteration)
        scouts_entry = ttk.Entry(param_frame2, textvariable=scouts)
        perspective_b_entry = ttk.Entry(param_frame2, textvariable=perspective_b)
        best_b_entry = ttk.Entry(param_frame2, textvariable=best_b)
        perspective_a_entry = ttk.Entry(param_frame2, textvariable=perspective_a)
        best_a_entry = ttk.Entry(param_frame2, textvariable=best_a)
        size_a_entry = ttk.Entry(param_frame2, textvariable=size_a)
        stop_entry = ttk.Entry(param_frame2, textvariable=stop)

        iteration_entry.grid(row=1, column=1)
        scouts_entry.grid(row=2, column=1)
        perspective_b_entry.grid(row=3, column=1)
        best_b_entry.grid(row=4, column=1)
        perspective_a_entry.grid(row=5, column=1)
        best_a_entry.grid(row=6, column=1)
        size_a_entry.grid(row=7, column=1)
        stop_entry.grid(row=8, column=1)


        separator = ttk.Separator(param_frame2, orient="horizontal")  # Горизонтальная полоса разделения
        separator.grid(row=9, column=0, columnspan=2, sticky="ew", pady=10)

        # Параметры функции
        ttk.Label(param_frame2, text="Функция и отображение ее графика", font=("Helvetica", 12,"bold")).grid(row=9, column=0, pady=10)
        ttk.Label(param_frame2, text="Выберите функцию", font=("Helvetica", 10)).grid(row=10, column=0)
        function_choices = ["Нажмите для выбора","Функция Химмельблау", "Функция Розенброка",
                        "Функция Растригина"]
        function_var = tk.StringVar(value=function_choices[0])
        function_menu = ttk.Combobox(param_frame2, textvariable=function_var, values=function_choices, width=22)
        function_menu.grid(row=10, column=1, pady=5)

        # Создание кнопки Выполнить
        button_style = ttk.Style()
        button_style.configure("My.TButton", font=("Helvetica", 14))

        # Создание кнопки Выполнить
        apply_settings_button = ttk.Button(param_frame2, text="Выполнить",command=run_optimization, style="My.TButton")
        apply_settings_button.grid(row=14, column=1, padx=10, pady=10)

        ttk.Label(param_frame2, text="Выполнение и результаты", font=("Helvetica", 12,"bold")).grid(row=11, column=0, pady=10)
        results_text = scrolledtext.ScrolledText(param_frame2, wrap=tk.WORD, height=16, width=40, padx=2, state=tk.DISABLED)
        results_text.grid(row=14, column=0, padx=10)
        