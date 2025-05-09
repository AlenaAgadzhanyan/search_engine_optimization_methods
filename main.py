import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext, messagebox
from tkinter.ttk import Notebook, Combobox
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from gradient import *
from quadratic_programming import *
from functions import *
from swarm import *
from bee import *
from immune import *
from bacteria import *
from hybrid import *
import numpy as np

# Функция для отрисовки поверхности
def plot_surface(ax, selected_function):
    ax.cla()
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    x, y = np.meshgrid(x, y)
    if selected_function == "Функция Химмельблау":
        z = np.vectorize(lambda x, y: (x**2 + y - 11)**2 + (x + y**2 - 7)**2)(x, y)
    elif selected_function == "Функция Бута":
        z = np.vectorize(lambda x, y: (x + 2*y - 7)**2 + (2*x + y - 5)**2)(x, y)
    elif selected_function == "Функция Сферы":
        z = np.vectorize(lambda x, y: x**2 + y**2)(x, y)
    elif selected_function == "Обратная функция Сферы":
        x = np.linspace(-5.5, 5.5, 100)
        y = np.linspace(-5.5, 5.5, 100)
        x, y = np.meshgrid(x, y)
        z = np.vectorize(lambda x, y: -(x**2 + y**2))(x, y)
    elif selected_function == "2x^2+2xy+2y^2-4x-6y":
        z = np.vectorize(lambda x, y: 2*x**2 + 2*x*y + 2*y**2 - 4*x - 6*y)(x, y)
    elif selected_function == "Функция Растригина":
        z = np.vectorize(lambda x, y: x**2 - 10 * np.cos(2 * np.pi * x) + y**2 - 10 * np.cos(2 * np.pi * y) + 20) (x,y)
    elif selected_function == "Функция Розенброка":
        x = np.linspace(-2, 2, 100)
        y = np.linspace(-1, 3, 100)
        x, y = np.meshgrid(x, y)
        z = np.vectorize(lambda x, y: sum(100 * (b - a**2)**2 + (1 - a)**2 for a, b in zip([x], [y])))(x, y)

    ax.plot_surface(x, y, z, rstride=5, cstride=5, alpha=0.5, cmap="inferno")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

def main():
    window = tk.Tk()
    width = window.winfo_screenwidth()
    height = window.winfo_screenheight()
    window.geometry("%dx%d" % (width, height))
    window.title("Оптимизация")

    # Меню
    menu_frame = tk.Frame(window, width=300, bg="#008080")
    menu_frame.pack(side=tk.LEFT, fill=tk.Y)

    # Для графика
    plot_frame = tk.Frame(window, bg="#008080")
    plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    # График
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # Панель инструментов для графика
    toolbar = NavigationToolbar2Tk(canvas, plot_frame)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # Функция для выполнения градиентного спуска
    def run_gradient_descent():
        try:
            selected_function = combobox_functions.get()
            x_start = float(entry_x.get())
            y_start = float(entry_y.get())
            step_size = float(entry_step.get())
            max_iterations = int(entry_iterations.get())
            delay = float(entry_delay.get())

            history = gradient_descent(x_start, y_start, step_size, max_iterations, selected_function)

            plot_surface(ax, selected_function)
            for i, (x, y, z) in enumerate(history):
                ax.scatter(x, y, z, c="black", s=10, marker="o")
                if i == len(history) - 1 and len(history) - 1 < max_iterations:
                    ax.scatter(x, y, z, c="red", s=50, marker="x")
                canvas.draw()
                output_text.insert(tk.END, f"{i + 1}) ({x:.2f}, {y:.2f}) = {z:.4f}\n")
                output_text.yview_moveto(1)
                window.update()
                window.after(int(delay * 1000))

            if len(history)-1 == max_iterations:
                messagebox.showwarning("Предупреждение", "Достигнуто максимальное количество итераций, возможно, не найден минимум.")
            else: messagebox.showinfo("Уведомление", "Градиентный спуск завершен!")
        except ValueError:
            messagebox.showerror("Ошибка", "Пожалуйста, введите корректные значения.")

    # Функция для выполнения симплекс-метода
    def run_simplex_method():
        try:
            output_text_simplex.delete("1.0", tk.END)
            output_text.delete("1.0", tk.END)

            delay = float(delay_simplex.get())

            # Запускаем симплекс-метод
            points = list(simplex_method())

            if not points:
                messagebox.showinfo("Уведомление", "Метод не нашёл решения.")
                return
            plot_surface(ax, "2x^2+2xy+2y^2-4x-6y")
            # Получаем последний результат
            result = points[1]
            point = points[0][0:2]
            yy = points[2]
            for i, (x, y, z) in enumerate(yy):
                ax.scatter(x, y, z, c="black", s=10, marker="o")
                if i == len(yy) - 1:
                    ax.scatter(x, y, z, c="red", s=50, marker="x")
                canvas.draw()
                output_text_simplex.insert(tk.END, f"{i + 1}) ({x:.2f}, {y:.2f}) = {z:.4f}\n")
                output_text_simplex.yview_moveto(1)
                window.update()
                window.after(int(delay * 1000))

            output_text_simplex.insert(tk.END, f"Точки: {point}\nЗначение функции: {result}\n")
            messagebox.showinfo("Уведомление", "Симплекс-метод завершен!")

        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    # Логика для роя чатастиц
    def execute():
        results_text.delete("1.0", tk.END)
        dim = 3
        swarm_size = int(particle.get())
        iter_count = int(iteration.get())
        alpha_val = float(alpha.get())
        beta_val = float(beta.get())
        inertia_val = float(inertia.get())
        min_val, max_val = -5.12, 5.12

        positions, velocities, best_local_positions, best_local_scores = initialize_swarm(swarm_size, dim, min_val, max_val)
        global_best_idx = np.argmin(best_local_scores)
        global_best_position = best_local_positions[global_best_idx]
        global_best_score = best_local_scores[global_best_idx]

        results_text.config(state=tk.NORMAL)
        results_text.delete(1.0, tk.END)

        ax.cla()
        plot_surface(ax, "Функция Растригина")
        xs = positions[:, 0]
        ys = positions[:, 1]
        zs = np.array([rastrigin(pos[:2]) for pos in positions])
        scatter = ax.scatter(xs, ys, zs, color='red', s=10)
        ax.set_title("Итерация 0")
        canvas.draw()
        window.update()

        for i in range(iter_count):
            ax.set_title(f"Итерация {i}")
            results_text.insert(tk.END, f"Итерация {i}\n")
            results_text.insert(tk.END, f"Позиция: {positions[0]}\n")
            results_text.insert(tk.END, f"Скорость: {velocities[0]}\n")
            results_text.insert(tk.END, f"Глобальный минимум: {global_best_position} (значение: {global_best_score})\n\n")

            positions, velocities, best_local_positions, best_local_scores, global_best_position, global_best_score = update_swarm(
                positions, velocities, best_local_positions, best_local_scores,
                global_best_position, inertia_val, alpha_val, beta_val, min_val, max_val)
            
            xs = positions[:, 0]
            ys = positions[:, 1]
            zs = np.array([rastrigin(pos[:2]) for pos in positions])
            scatter._offsets3d = (xs, ys, zs)
            ax.set_title(f"Итерация {i}")
            
            results_text.yview_moveto(1)
            canvas.draw()
            window.update()

    # Пчелы
    def optimize_bees(num_scouts, elite_radius, perspective_radius,
                    num_elite_bees, num_perspective_bees,
                    num_elite_sites, num_perspective_sites,
                    max_epochs, bounds,
                    # stagnation_limit,
                    fitness_func, result_widget, func):
        
        # 1. Инициализация популяции
        scouts = initialize_bees(num_scouts, fitness_func, bounds)
        best_solution = min(scouts, key=lambda x: x[1])
        best_fitness = best_solution[1]
        stagnation = 0

        # 2. Визуализация начального состояния
        ax.clear()
        plot_surface(ax, func)
        scatter = ax.scatter([b[0][0] for b in scouts], [b[0][1] for b in scouts], 
                            [b[1] for b in scouts], color='red', s=20, label='Разведчики')
        canvas.draw()
        window.update()

        for epoch in range(max_epochs):
            # 3. Фаза разведки - сортируем и выбираем лучшие участки
            scouts.sort(key=lambda b: b[1])
            
            # Разделяем на элитные и перспективные участки
            elite_sites = scouts[:num_elite_sites]
            perspective_sites = scouts[num_elite_sites:num_elite_sites + num_perspective_sites]
            
            # 4. Фаза работы - отправляем рабочих пчел
            workers = []
            
            # На элитные участки отправляем больше рабочих
            for site in elite_sites:
                for _ in range(num_elite_bees):
                    worker = explore_bee(site, elite_sites, elite_radius, perspective_radius, fitness_func)
                    workers.append(worker)
            
            # На перспективные участки отправляем меньше рабочих
            for site in perspective_sites:
                for _ in range(num_perspective_bees):
                    worker = explore_bee(site, perspective_sites, elite_radius, perspective_radius, fitness_func)
                    workers.append(worker)
            
            # 5. Добавляем новых разведчиков (случайный поиск)
            new_scouts = initialize_bees(num_scouts, fitness_func, bounds)
            
            # 6. Объединяем всех пчел
            all_bees = new_scouts + workers
            
            # 7. Отбор новых разведчиков
            all_bees.sort(key=lambda b: b[1])
            new_scouts = all_bees[:num_scouts]
            
            # 8. Проверка на улучшение
            current_best = min(new_scouts, key=lambda x: x[1])
            if current_best[1] < best_fitness:
                best_solution = current_best
                best_fitness = current_best[1]
                stagnation = 0
            else:
                stagnation += 1

            scouts = all_bees

            xs = [b[0][0] for b in scouts]
            ys = [b[0][1] for b in scouts]
            zs = [b[1] for b in scouts]
            scatter._offsets3d = (xs, ys, zs)
            ax.set_title(f"Итерация {epoch}: Лучшее значение {best_fitness:.4f}")
            result_widget.insert(tk.END, 
                f"Итерация {epoch}: Лучшее ({best_solution[0][0]:.4f}, {best_solution[0][1]:.4f}) -> {best_solution[1]:.4f}\n")
            result_widget.yview_moveto(1)
            canvas.draw()
            window.update()

        return best_solution

    def run_bee_algorithm():
        ax.cla()
        result_widget.delete("1.0", tk.END)
        func = function_bee.get()

        num_scouts = int(num_scouts_entry.get())  # Разведчики
        num_elite_sites = int(num_elite_sites_entry.get())  # Кол-во элитных участков
        num_perspective_sites = int(num_perspective_sites_entry.get())  # Кол-во перспективных участков
        num_elite_bees = int(num_elite_entry.get())  # Рабочих пчёл на элитный участок
        num_perspective_bees = int(num_perspective_entry.get())  # Рабочих пчёл на перспективный участок

        max_epochs = int(max_epochs_entry.get())

        if func == "Функция Химмельблау":
            f = himmelblau
            elite_radius = 0.3
            perspective_radius = 1.5
            bounds = [(-5.12, 5.12), (-5.12, 5.12)]
        elif func == "Функция Растригина":
            f = rastrigin
            elite_radius = 0.1
            perspective_radius = 1.0
            bounds = [(-5.12, 5.12), (-5.12, 5.12)]
        elif func == "Функция Розенброка":
            f = rosenbrock
            elite_radius = 0.1
            perspective_radius = 0.4
            bounds = [(-2, 2), (-1, 3)]
        else:
            result_widget.insert(tk.END, "Не выбрана функция!\n")
            return

        best = optimize_bees(num_scouts, elite_radius, perspective_radius, num_elite_bees, num_perspective_bees, num_elite_sites,
            num_perspective_sites, max_epochs, bounds,
            f, result_widget, func)

        result_widget.insert(tk.END, f"\nЛучшее решение: ({best[0][0]:.4f}, {best[0][1]:.4f}) -> {best[1]:.4f}\n")
        result_widget.yview_moveto(1)

    # Имунный алгоритм
    def draw_lab_6():
        txt_f_tab_6.delete("1.0", tk.END)
        
        pop_number = int(size6.get())
        iter_number = int(it.get())
        clon = int(clones.get())
        best_clon = int(resClones6.get())
        best_pop = int(res6.get())
        
        plot_surface(ax, "Функция Розенброка") 

        func = rosenbrock
        bounds = [(-2, 2), (-1, 3)]

        agents = init_population(func, pop_number, bounds)

        points = ax.scatter([ag[0] for ag in agents],
                            [ag[1] for ag in agents],
                            [ag[2] for ag in agents],
                            c="red", s=20, marker="o")

        best_agent = get_best(agents)
        
        canvas.draw()
        window.update()
        best_point = None
        for i in range(iter_number):
            agents = immune_step(agents, func, best_pop, clon, best_clon, 1 / (i + 1), bounds)

            points.remove()
            points = ax.scatter([ag[0] for ag in agents],
                                [ag[1] for ag in agents],
                                [ag[2] for ag in agents],
                                c="red", s=20, marker="o")

            best_agent = get_best(agents)
            if best_point is not None:
                best_point.remove()
            best_point = ax.scatter(best_agent[0], best_agent[1], best_agent[2], c="blue", s=60, marker="o")

            txt_f_tab_6.insert(tk.END,
                f"{i + 1}) ({best_agent[0]:.8f}) "
                f"({best_agent[1]:.8f}) = "
                f"({best_agent[2]:.8f})\n")
            txt_f_tab_6.yview_moveto(1)

            canvas.draw()
            window.update()

        for ag in agents:
            ax.scatter(ag[0], ag[1], ag[2], c="red", s=20, marker="o")
        
        ax.scatter(best_agent[0], best_agent[1], best_agent[2], c="blue", s=80, marker="x")
        
        txt_f_tab_6.insert(tk.END, f"\nЛучшее найденное решение: \n({best_agent[0]:.8f}) ({best_agent[1]:.8f}) = ({best_agent[2]:.8f})")
        txt_f_tab_6.yview_moveto(1)

        canvas.draw()
        window.update()
        messagebox.showinfo('Уведомление', 'Готово')


    def draw_bfo():
        txt_f_tab_7.delete("1.0", tk.END)

        pop_number = int(size7.get())      # количество агентов
        iter_number = int(it7.get())       # число итераций
        swim_steps = int(swim7.get())      # шаги плавания
        elim_step = float(elim7.get())     # шаг ликвидации
        elim_num = int(elim_num7.get())    # процент ликвидируемых
        func = inverse_sphere

        if pop_number % 2 != 0:
            messagebox.showerror("Ошибка", "Количество агентов должно быть чётным!")
            return

        ax.cla()
        plot_surface(ax, "Обратная функция Сферы")

        all_positions = bacteria_foraging_optimization(
            n=pop_number,
            function=func,
            iteration=iter_number,
            num_swimming_steps=swim_steps,
            step_size=elim_step,
            dispersion_probability=elim_num
        )

        for t, agents in enumerate(all_positions):
            z_vals = [func(x) for x in agents]
            ax.cla()
            plot_surface(ax, "Обратная функция Сферы")

            ax.scatter([x[0] for x in agents],
                    [x[1] for x in agents],
                    z_vals,
                    c="green", s=20, marker="o")

            best_idx = np.argmax(z_vals)
            best_agent = agents[best_idx]
            best_val = z_vals[best_idx]

            ax.scatter(best_agent[0], best_agent[1], best_val, c="red", s=100, marker="o")
            
            txt_f_tab_7.insert(tk.END,
                f"{t}) ({best_agent[0]:.8f}) "
                f"({best_agent[1]:.8f}) = "
                f"({best_val:.8f})\n")
            txt_f_tab_7.yview_moveto(1)

            canvas.draw()
            window.update()

        ax.scatter(best_agent[0], best_agent[1], best_val, c="red", s=40, marker="x")
        
        txt_f_tab_7.insert(tk.END, f"\nЛучшее найденное решение: \n({best_agent[0]:.8f}) ({best_agent[1]:.8f}) = ({best_val:.8f})")
        txt_f_tab_7.yview_moveto(1)

        canvas.draw()
        window.update()
        messagebox.showinfo('Уведомление', 'Готово')

    def draw_hybrid():
        txt_f_tab_8.delete("1.0", tk.END)

        pop_number = int(size8.get())      # количество агентов
        iter_number = int(it8.get())       # число итераций
        swim_steps = int(swim8.get())      # шаги плавания
        elim_step = float(elim8.get())     # шаг ликвидации
        elim_num = int(elim_num8.get())    # процент ликвидируемых
        selected_func = combobox_functions8.get()   # используемая функция

        # === Параметры иммунного алгоритма ===
        best_numb = int(best_numb8.get())
        clon_numb = int(clon_numb8.get())
        best_clon_numb = int(best_clon_numb8.get())

        if pop_number % 2 != 0:
            messagebox.showerror("Ошибка", "Количество агентов должно быть чётным!")
            return
        
        if selected_func == "Функция Растригина":
            func = rastrigin
            maxi = False
        elif selected_func == "Обратная функция Сферы":
            func = inverse_sphere
            maxi = True

        ax.cla()
        plot_surface(ax, selected_func)

        bounds = [(-5, 5), (-5, 5)]

        all_positions, best_agent = hybrid_algorithm(
            func=func,
            n=pop_number,
            iterations=iter_number,
            swim_steps=swim_steps,
            step_size=elim_step,
            dispersion_prob=elim_num,
            best_numb=best_numb,
            clon_numb=clon_numb,
            best_clon_numb=best_clon_numb,
            bounds=bounds,
            maximize=maxi
        )

        for t, agents in enumerate(all_positions):
            z_vals = [func(x) for x in agents]
            ax.cla()
            plot_surface(ax, selected_func)

            ax.scatter([x[0] for x in agents],
                    [x[1] for x in agents],
                    z_vals,
                    c="blue", s=20, marker="o")

            if (maxi):
                best_idx = np.argmax(z_vals)
            else:
                best_idx = np.argmin(z_vals)
            best_agent = agents[best_idx]
            best_val = z_vals[best_idx]

            ax.scatter(best_agent[0], best_agent[1], best_val, c="red", s=100, marker="o")
            
            txt_f_tab_8.insert(tk.END,
                f"{t}) ({best_agent[0]:.8f}) "
                f"({best_agent[1]:.8f}) = "
                f"({best_val:.8f})\n")
            txt_f_tab_8.yview_moveto(1)

            canvas.draw()
            window.update()

        ax.scatter(best_agent[0], best_agent[1], best_val, c="red", s=40, marker="x")
        
        txt_f_tab_8.insert(tk.END, f"\nЛучшее найденное решение: \n({best_agent[0]:.8f}) ({best_agent[1]:.8f}) = ({best_val:.8f})")
        txt_f_tab_8.yview_moveto(1)

        canvas.draw()
        window.update()
        messagebox.showinfo('Уведомление', 'Готово')



    def update_plot_simplex(event=None):
        current_tab = tab_control.index(tab_control.select())
        if current_tab == 0:  # LR1 (Градиентный спуск)
            plot_surface(ax, "Функция Химмельблау")
        elif current_tab == 1:  # LR2 (Симплекс-метод)
            plot_surface(ax, "2x^2+2xy+2y^2-4x-6y")
        elif current_tab == 3 or current_tab == 4:  # LR4 (Рой частиц)
            plot_surface(ax, "Функция Растригина")
        elif current_tab == 5:  # LR6
            plot_surface(ax, "Функция Розенброка")
        elif current_tab == 6 or current_tab == 7:  # LR6
            plot_surface(ax, "Обратная функция Сферы")
        canvas.draw()

    # Функция для очистки вывода
    def clear_output():
        output_text.delete(1.0, tk.END)
        output_text_simplex.delete(1.0, tk.END)
        results_text.delete(1.0, tk.END)
        result_widget.delete(1.0, tk.END)


    # Функция для обновления графика при выборе функции
    def update_plot(event=None):
        current_tab = tab_control.index(tab_control.select())
        if current_tab == 0:
            selected_function = combobox_functions
        elif current_tab == 4:
            selected_function = function_bee.get()
        elif current_tab == 6:
            selected_function = inverse_sphere
        elif current_tab == 7:
            selected_function = combobox_functions8.get()
        plot_surface(ax, selected_function)
        canvas.draw()

    # Создание вкладок в меню
    tab_control = Notebook(menu_frame)

    # Вкладка для градиентного спуска
    tab1 = tk.Frame(tab_control, bg="#008080")
    tab_control.add(tab1, text="LR1")

    label_title = tk.Label(tab1, text="Градиентный спуск", font=("Arial", 18, "bold"), fg="#FFFFFF", bg="#008080")
    label_title.pack(pady=10)

    label_function = tk.Label(tab1, text="Выберите функцию:", bg="#008080", font=("Arial", 16), fg="#FFFFFF")
    label_function.pack(pady=5)
    combobox_functions = Combobox(tab1, values=["Функция Химмельблау", "Функция Сферы", "Функция Бута"])
    combobox_functions.pack(pady=5)
    combobox_functions.set("Функция Химмельблау")
    combobox_functions.bind("<<ComboboxSelected>>", update_plot)

    label_x = tk.Label(tab1, text="Начальное значение x:", bg="#008080", font=("Arial", 16), fg="#FFFFFF")
    label_x.pack(pady=5)
    entry_x = tk.Entry(tab1, bg="#E0FFFF")
    entry_x.insert(0, "0")
    entry_x.pack(pady=5)

    label_y = tk.Label(tab1, text="Начальное значение y:", bg="#008080", font=("Arial", 16), fg="#FFFFFF")
    label_y.pack(pady=5)
    entry_y = tk.Entry(tab1, bg="#E0FFFF")
    entry_y.insert(0, "0")
    entry_y.pack(pady=5)

    label_step = tk.Label(tab1, text="Шаг (α):", bg="#008080", font=("Arial", 16), fg="#FFFFFF")
    label_step.pack(pady=5)
    entry_step = tk.Entry(tab1, bg="#E0FFFF")
    entry_step.insert(0, "0.01")
    entry_step.pack(pady=5)

    label_iterations = tk.Label(tab1, text="Макс. кол-во итераций:", bg="#008080", font=("Arial", 16), fg="#FFFFFF")
    label_iterations.pack(pady=5)
    entry_iterations = tk.Entry(tab1, bg="#E0FFFF")
    entry_iterations.insert(0, "1000")
    entry_iterations.pack(pady=5)

    label_delay = tk.Label(tab1, text="Задержка между итерациями (сек):", bg="#008080", font=("Arial", 16), fg="#FFFFFF")
    label_delay.pack(pady=5)
    entry_delay = tk.Entry(tab1, bg="#E0FFFF")
    entry_delay.insert(0, "0.001")
    entry_delay.pack(pady=5)

    button_frame = tk.Frame(tab1, bg="#008080")
    button_frame.pack(side=tk.TOP, padx=10)

    # Кнопки для запуска и очистки
    run_button = tk.Button(button_frame, text="Запуск", font=("Arial", 12, "bold"), fg="#008080", width=15, height=1, command=run_gradient_descent)
    run_button.pack(side=tk.LEFT, padx=10)

    clear_button = tk.Button(button_frame, text="Очистить вывод", font=("Arial", 12, "bold"), fg="#008080", width=15, height=1, command=clear_output)
    clear_button.pack(side=tk.RIGHT,pady=10)

    # Вывод результатов
    label_delay = tk.Label(tab1, text="Результаты:", bg="#008080", font=("Arial", 16), fg="#FFFFFF")
    label_delay.pack(pady=1)
    output_text = scrolledtext.ScrolledText(tab1, width=35, bg="#E0FFFF")
    output_text.pack(fill=tk.Y, pady=5)

    tab_control.bind("<<NotebookTabChanged>>", update_plot_simplex)

    # Вкладка для симплекс-метода
    tab2 = tk.Frame(tab_control, bg="#008080")
    tab_control.add(tab2, text="LR2")

    label_title_simplex = tk.Label(tab2, text="Симплекс-метод", font=("Arial", 18, "bold"), fg="#FFFFFF", bg="#008080")
    label_title_simplex.pack(pady=10)
    label_delay_simplex = tk.Label(tab2, text="Задержка между итерациями (сек):", bg="#008080", font=("Arial", 16), fg="#FFFFFF")
    label_delay_simplex.pack(pady=5)
    delay_simplex = tk.Entry(tab2, bg="#E0FFFF")
    delay_simplex.insert(0, "0.5")
    delay_simplex.pack(pady=5)

    button_frame_simplex = tk.Frame(tab2, bg="#008080")
    button_frame_simplex.pack(side=tk.TOP, padx=10)

    run_button_simplex = tk.Button(button_frame_simplex, text="Запуск", font=("Arial", 12, "bold"), fg="#008080", width=15, height=1, command=run_simplex_method)
    run_button_simplex.pack(side=tk.LEFT, padx=10)

    clear_button_simplex = tk.Button(button_frame_simplex, text="Очистить вывод", font=("Arial", 12, "bold"), fg="#008080", width=15, height=1, command=clear_output)
    clear_button_simplex.pack(side=tk.RIGHT,pady=10)

    label_delay_simplex = tk.Label(tab2, text="Результаты:", bg="#008080", font=("Arial", 16), fg="#FFFFFF")
    label_delay_simplex.pack(pady=1)
    output_text_simplex = scrolledtext.ScrolledText(tab2, width=35, bg="#E0FFFF")
    output_text_simplex.pack(fill=tk.Y, pady=5)

    tab_control.pack(expand=1, fill="both")

    tab3 = tk.Frame(tab_control, bg="#008080")
    tab_control.add(tab3, text="LR3")

    # Рой частиц
    tab4 = tk.Frame(tab_control, bg="#008080")
    tab_control.add(tab4, text="LR4")

    label_title4 = tk.Label(tab4, text="Алгоритм роя частиц", font=("Arial", 18, "bold"), fg="#FFFFFF", bg="#008080")
    label_title4.pack(pady=10)

    particle = tk.IntVar(value=500)
    iteration = tk.IntVar(value=100)
    alpha = tk.DoubleVar(value=0.8)
    beta = tk.DoubleVar(value=0.9)
    inertia = tk.DoubleVar(value=0.5)

    params = [("Частицы", particle), ("Итерации", iteration), ("Альфа", alpha), ("Бета", beta), ("Инерция", inertia)]
    param_frame = tk.Frame(tab4, bg="#008080")
    param_frame.pack(pady=10)

    for idx, (label, var) in enumerate(params):
        ttk.Label(param_frame, text=label).grid(row=idx, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(param_frame, textvariable=var).grid(row=idx, column=1, padx=5, pady=5)

    button_frame = tk.Frame(tab4, bg="#008080")
    button_frame.pack(pady=10)

    style = ttk.Style()
    style.configure("Custom.TButton",
                font=("Arial", 12, "bold"),
                foreground="#008080",
                width=15, height=1)

    ttk.Button(button_frame, text="Запуск", style="Custom.TButton", command=execute).pack(side="left", padx=5)
    ttk.Button(button_frame, text="Очистить вывод", style="Custom.TButton", command=clear_output).pack(side="left", padx=5)

    ttk.Label(tab4, text="Результаты", font=("Helvetica", 12, "bold"), background="#008080", foreground="#FFFFFF").pack(pady=5)
    results_text = scrolledtext.ScrolledText(tab4, wrap=tk.WORD, height=18, width=60, padx=10, state=tk.DISABLED)
    results_text.pack(padx=10, pady=5)

    # Пчелинный рой
    tab5 = tk.Frame(tab_control, bg="#008080")
    tab_control.add(tab5, text="LR5")

    label_title5 = tk.Label(tab5, text="Пчелинный алгоритм", font=("Arial", 18, "bold"), fg="#FFFFFF", bg="#008080")
    label_title5.pack(pady=10)

    function_bee = ttk.Combobox(tab5, values=["Функция Растригина", "Функция Химмельблау", "Функция Розенброка"])
    function_bee.set("Функция Растригина")
    function_bee.pack()
    function_bee.bind("<<ComboboxSelected>>", update_plot)

    input_frame = tk.Frame(tab5, bg="#008080")
    input_frame.pack(pady=10)

    # Фрейм слева для лейблов
    labels_frame = tk.Frame(input_frame, bg="#008080")
    labels_frame.pack(side=tk.LEFT, padx=10)

    # Фрейм справа для полей ввода
    entries_frame = tk.Frame(input_frame, bg="#008080")
    entries_frame.pack(side=tk.LEFT, padx=10)

    label_iterations = tk.Label(labels_frame, text="Итерации:", bg="#008080", font=("Arial", 12), fg="#FFFFFF")
    label_iterations.pack(pady=2.5)
    max_epochs_entry = tk.Entry(entries_frame)
    max_epochs_entry.insert(0, "100")
    max_epochs_entry.pack(padx=5, pady=5)

    label_scouts = tk.Label(labels_frame, text="Разведчики:", bg="#008080", font=("Arial", 12), fg="#FFFFFF")
    label_scouts.pack(pady=2.5)
    num_scouts_entry = tk.Entry(entries_frame)
    num_scouts_entry.insert(0, "20")
    num_scouts_entry.pack(padx=5, pady=5)

    label_persp_bees = tk.Label(labels_frame, text="Пчёл в перспективном участке:", bg="#008080", font=("Arial", 12), fg="#FFFFFF")
    label_persp_bees.pack(pady=2.5)
    num_perspective_entry = tk.Entry(entries_frame)
    num_perspective_entry.insert(0, "10")
    num_perspective_entry.pack(padx=5, pady=5)

    label_elite_bees = tk.Label(labels_frame, text="Пчёл в лучшем участке:", bg="#008080", font=("Arial", 12), fg="#FFFFFF")
    label_elite_bees.pack(pady=2.5)
    num_elite_entry = tk.Entry(entries_frame)
    num_elite_entry.insert(0, "20")
    num_elite_entry.pack(padx=5, pady=5)

    label_persp_sites = tk.Label(labels_frame, text="Перспективных участков:", bg="#008080", font=("Arial", 12), fg="#FFFFFF")
    label_persp_sites.pack(pady=2.5)
    num_perspective_sites_entry = tk.Entry(entries_frame)
    num_perspective_sites_entry.insert(0, "3")
    num_perspective_sites_entry.pack(padx=5, pady=5)

    label_elite_sites = tk.Label(labels_frame, text="Лучших участков:", bg="#008080", font=("Arial", 12), fg="#FFFFFF")
    label_elite_sites.pack(pady=2.5)
    num_elite_sites_entry = tk.Entry(entries_frame)
    num_elite_sites_entry.insert(0, "1")
    num_elite_sites_entry.pack(padx=5, pady=5)

    button_bee = tk.Frame(tab5, bg="#008080")
    button_bee.pack(pady=10)

    ttk.Button(button_bee, text="Запуск пчёл", style="Custom.TButton", command=run_bee_algorithm).pack(side="left", padx=5)
    ttk.Button(button_bee, text="Очистить вывод", style="Custom.TButton", command=clear_output).pack(side="left", padx=5)

    ttk.Label(tab5, text="Результаты", font=("Helvetica", 12, "bold"), background="#008080", foreground="#FFFFFF").pack(pady=5)
    result_widget = scrolledtext.ScrolledText(tab5, width=60, height=20)
    result_widget.pack()

    # Имунный алгоритм
    tab6 = tk.Frame(tab_control, bg="#008080")
    tab_control.add(tab6, text="LR6")

    label_title5 = tk.Label(tab6, text="Иммунный алгоритм", font=("Arial", 18, "bold"), fg="#FFFFFF", bg="#008080")
    label_title5.pack(pady=10)

    input_frame6 = tk.Frame(tab6, bg="#008080")
    input_frame6.pack(pady=10)

    # Фрейм слева для лейблов
    labels_frame6 = tk.Frame(input_frame6, bg="#008080")
    labels_frame6.pack(side=tk.LEFT, padx=10)

    # Фрейм справа для полей ввода
    entries_frame6 = tk.Frame(input_frame6, bg="#008080")
    entries_frame6.pack(side=tk.LEFT, padx=10)

    label_iterations6 = tk.Label(labels_frame6, text="Итерации:", bg="#008080", font=("Arial", 12), fg="#FFFFFF")
    label_iterations6.pack(pady=2.5)
    it = tk.Entry(entries_frame6)
    it.insert(0, "100")
    it.pack(padx=5, pady=5)

    label_62 = tk.Label(labels_frame6, text="Размер популяции:", bg="#008080", font=("Arial", 12), fg="#FFFFFF")
    label_62.pack(pady=2.5)
    size6 = tk.Entry(entries_frame6)
    size6.insert(0, "50")
    size6.pack(padx=5, pady=5)

    label_63 = tk.Label(labels_frame6, text="Кол-во клонов:", bg="#008080", font=("Arial", 12), fg="#FFFFFF")
    label_63.pack(pady=2.5)
    clones = tk.Entry(entries_frame6)
    clones.insert(0, "20")
    clones.pack(padx=5, pady=5)

    label_64 = tk.Label(labels_frame6, text="Кол-во лучших решений из популяции:", bg="#008080", font=("Arial", 12), fg="#FFFFFF")
    label_64.pack(pady=2.5)
    res6 = tk.Entry(entries_frame6)
    res6.insert(0, "10")
    res6.pack(padx=5, pady=5)

    label_65 = tk.Label(labels_frame6, text="Кол-во лучших решений из клонов:", bg="#008080", font=("Arial", 12), fg="#FFFFFF")
    label_65.pack(pady=2.5)
    resClones6 = tk.Entry(entries_frame6)
    resClones6.insert(0, "10")
    resClones6.pack(padx=5, pady=5)

    button_im = tk.Frame(tab6, bg="#008080")
    button_im.pack(pady=10)

    ttk.Button(button_im, text="Запуск", style="Custom.TButton", command=draw_lab_6).pack(side="left", padx=5)
    ttk.Button(button_im, text="Очистить вывод", style="Custom.TButton", command=lambda: txt_f_tab_6.delete(1.0, tk.END)).pack(side="left", padx=5)

    ttk.Label(tab6, text="Результаты", font=("Helvetica", 12, "bold"), background="#008080", foreground="#FFFFFF").pack(pady=5)
    txt_f_tab_6 = scrolledtext.ScrolledText(tab6, width=60, height=20)
    txt_f_tab_6.pack()

   # Бактериальный алгоритм
    tab7 = tk.Frame(tab_control, bg="#008080")
    tab_control.add(tab7, text="LR7")

    label_title71 = tk.Label(tab7, text="Бактериальный алгоритм", font=("Arial", 18, "bold"), fg="#FFFFFF", bg="#008080")
    label_title71.pack(pady=10)

    input_frame7 = tk.Frame(tab7, bg="#008080")
    input_frame7.pack(pady=10)

    # Фрейм слева для лейблов
    labels_frame7 = tk.Frame(input_frame7, bg="#008080")
    labels_frame7.pack(side=tk.LEFT, padx=10)

    # Фрейм справа для полей ввода
    entries_frame7 = tk.Frame(input_frame7, bg="#008080")
    entries_frame7.pack(side=tk.LEFT, padx=10)

    # Поля ввода параметров
    label_iterations7 = tk.Label(labels_frame7, text="Итерации:", bg="#008080", font=("Arial", 12), fg="#FFFFFF")
    label_iterations7.pack(pady=2.5)
    it7 = tk.Entry(entries_frame7)
    it7.insert(0, "100")
    it7.pack(padx=5, pady=5)

    label_size7 = tk.Label(labels_frame7, text="Размер популяции:", bg="#008080", font=("Arial", 12), fg="#FFFFFF")
    label_size7.pack(pady=2.5)
    size7 = tk.Entry(entries_frame7)
    size7.insert(0, "50")
    size7.pack(padx=5, pady=5)

    label_swim7 = tk.Label(labels_frame7, text="Шаги плавания:", bg="#008080", font=("Arial", 12), fg="#FFFFFF")
    label_swim7.pack(pady=2.5)
    swim7 = tk.Entry(entries_frame7)
    swim7.insert(0, "12")
    swim7.pack(padx=5, pady=5)

    label_elim7 = tk.Label(labels_frame7, text="Шаг ликвидации:", bg="#008080", font=("Arial", 12), fg="#FFFFFF")
    label_elim7.pack(pady=2.5)
    elim7 = tk.Entry(entries_frame7)
    elim7.insert(0, "0.2")
    elim7.pack(padx=5, pady=5)

    label_elim_num7 = tk.Label(labels_frame7, text="Процент ликвидируемых:", bg="#008080", font=("Arial", 12), fg="#FFFFFF")
    label_elim_num7.pack(pady=2.5)
    elim_num7 = tk.Entry(entries_frame7)
    elim_num7.insert(0, "25")
    elim_num7.pack(padx=5, pady=5)

    button_bact = tk.Frame(tab7, bg="#008080")
    button_bact.pack(pady=10)

    ttk.Button(button_bact, text="Запуск", style="Custom.TButton", command=draw_bfo).pack(side="left", padx=5)
    ttk.Button(button_bact, text="Очистить вывод", style="Custom.TButton", command=lambda: txt_f_tab_7.delete(1.0, tk.END)).pack(side="left", padx=5)

    ttk.Label(tab7, text="Результаты", font=("Helvetica", 12, "bold"), background="#008080", foreground="#FFFFFF").pack(pady=5)
    txt_f_tab_7 = scrolledtext.ScrolledText(tab7, width=60, height=20)
    txt_f_tab_7.pack()

    # Гибридный алгоритм
    tab8 = tk.Frame(tab_control, bg="#008080")
    tab_control.add(tab8, text="LR8")

    label_title81 = tk.Label(tab8, text="Гибридный алгоритм", font=("Arial", 18, "bold"), fg="#FFFFFF", bg="#008080")
    label_title81.pack(pady=10)

    label_function8 = tk.Label(tab8, text="Выберите функцию:", bg="#008080", font=("Arial", 16), fg="#FFFFFF")
    label_function8.pack(pady=5)
    combobox_functions8 = Combobox(tab8, values=["Обратная функция Сферы", "Функция Растригина"])
    combobox_functions8.pack(pady=5)
    combobox_functions8.set("Обратная функция Сферы")
    combobox_functions8.bind("<<ComboboxSelected>>", update_plot)


    input_frame8 = tk.Frame(tab8, bg="#008080")
    input_frame8.pack(pady=10)

    # Фрейм слева для лейблов
    labels_frame8 = tk.Frame(input_frame8, bg="#008080")
    labels_frame8.pack(side=tk.LEFT, padx=10)

    # Фрейм справа для полей ввода
    entries_frame8 = tk.Frame(input_frame8, bg="#008080")
    entries_frame8.pack(side=tk.LEFT, padx=10)

    # Поля ввода параметров
    label_iterations8 = tk.Label(labels_frame8, text="Итерации:", bg="#008080", font=("Arial", 12), fg="#FFFFFF")
    label_iterations8.pack(pady=2.5)
    it8 = tk.Entry(entries_frame8)
    it8.insert(0, "100")
    it8.pack(padx=5, pady=5)

    label_size8 = tk.Label(labels_frame8, text="Размер популяции:", bg="#008080", font=("Arial", 12), fg="#FFFFFF")
    label_size8.pack(pady=2.5)
    size8 = tk.Entry(entries_frame8)
    size8.insert(0, "50")
    size8.pack(padx=5, pady=5)

    label_swim8 = tk.Label(labels_frame8, text="Шаги плавания:", bg="#008080", font=("Arial", 12), fg="#FFFFFF")
    label_swim8.pack(pady=2.5)
    swim8 = tk.Entry(entries_frame8)
    swim8.insert(0, "12")
    swim8.pack(padx=5, pady=5)

    label_elim8 = tk.Label(labels_frame8, text="Шаг ликвидации:", bg="#008080", font=("Arial", 12), fg="#FFFFFF")
    label_elim8.pack(pady=2.5)
    elim8 = tk.Entry(entries_frame8)
    elim8.insert(0, "0.2")
    elim8.pack(padx=5, pady=5)

    label_elim_num8 = tk.Label(labels_frame8, text="Процент ликвидируемых:", bg="#008080", font=("Arial", 12), fg="#FFFFFF")
    label_elim_num8.pack(pady=2.5)
    elim_num8 = tk.Entry(entries_frame8)
    elim_num8.insert(0, "25")
    elim_num8.pack(padx=5, pady=5)

    label_83 = tk.Label(labels_frame8, text="Кол-во клонов:", bg="#008080", font=("Arial", 12), fg="#FFFFFF")
    label_83.pack(pady=2.5)
    clon_numb8 = tk.Entry(entries_frame8)
    clon_numb8.insert(0, "20")
    clon_numb8.pack(padx=5, pady=5)

    label_84 = tk.Label(labels_frame8, text="Кол-во лучших решений из популяции:", bg="#008080", font=("Arial", 12), fg="#FFFFFF")
    label_84.pack(pady=2.5)
    best_numb8 = tk.Entry(entries_frame8)
    best_numb8.insert(0, "10")
    best_numb8.pack(padx=5, pady=5)

    label_85 = tk.Label(labels_frame8, text="Кол-во лучших решений из клонов:", bg="#008080", font=("Arial", 12), fg="#FFFFFF")
    label_85.pack(pady=2.5)
    best_clon_numb8 = tk.Entry(entries_frame8)
    best_clon_numb8.insert(0, "10")
    best_clon_numb8.pack(padx=5, pady=5)

    button_hybrid = tk.Frame(tab8, bg="#008080")
    button_hybrid.pack(pady=10)

    ttk.Button(button_hybrid, text="Запуск", style="Custom.TButton", command=draw_hybrid).pack(side="left", padx=5)
    ttk.Button(button_hybrid, text="Очистить вывод", style="Custom.TButton", command=lambda: txt_f_tab_8.delete(1.0, tk.END)).pack(side="left", padx=5)

    ttk.Label(tab8, text="Результаты", font=("Helvetica", 12, "bold"), background="#008080", foreground="#FFFFFF").pack(pady=5)
    txt_f_tab_8 = scrolledtext.ScrolledText(tab8, width=60, height=20)
    txt_f_tab_8.pack()



    plot_surface(ax, "Функция Химмельблау")
    canvas.draw()

    window.mainloop()

if __name__ == "__main__":
    main()