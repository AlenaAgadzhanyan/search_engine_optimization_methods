import tkinter as tk
from tkinter import scrolledtext, messagebox
from tkinter.ttk import Notebook, Combobox
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from gradient import *
from quadratic_programming import *
from functions import *
from genetic_algorithm import *
from bee import *
from swarm import *

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
    elif selected_function == "2x^2+2xy+2y^2-4x-6y":
        z = np.vectorize(lambda x, y: 2*x**2 + 2*x*y + 2*y**2 - 4*x - 6*y)(x, y)

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


    def update_plot_simplex(event=None):
        current_tab = tab_control.index(tab_control.select())
        if current_tab == 0:  # LR1 (Градиентный спуск)
            plot_surface(ax, "Функция Химмельблау")
        elif current_tab == 1:  # LR2 (Симплекс-метод)
            plot_surface(ax, "2x^2+2xy+2y^2-4x-6y")
        canvas.draw()
    # Функция для очистки вывода
    def clear_output():
        output_text.delete(1.0, tk.END)
        output_text_simplex.delete(1.0, tk.END)
    # Функция для обновления графика при выборе функции
    def update_plot(event=None):
        selected_function = combobox_functions.get()
        plot_surface(ax, selected_function)
        canvas.draw()
    # Создание вкладок в меню
    tab_control = Notebook(menu_frame)
    # Вкладка для градиентного спуска
    tab1 = tk.Frame(tab_control, bg="#008080")
    tab_control.add(tab1, text="1")
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
    tab_control.add(tab2, text="2")
    label_title_simplex = tk.Label(tab2, text="Симплекс-метод", font=("Arial", 18, "bold"), fg="#FFFFFF", bg="#008080")
    label_title_simplex.pack(pady=10)
    label_delay_simplex = tk.Label(tab2, text="Задержка между итерациями (сек):", bg="#008080", font=("Arial", 16), fg="#FFFFFF")
    label_delay_simplex.pack(pady=5)
    delay_simplex = tk.Entry(tab2, bg="#E0FFFF")
    delay_simplex.insert(0, "0.5")
    delay_simplex.pack(pady=5)

    tab3 = tk.Frame(tab_control, bg="#008080")
    tab_control.add(tab3, text="3")
    GeneticAlgorithm(tab3,window,ax,canvas)

    tab4 = tk.Frame(tab_control, bg="#008080")
    tab_control.add(tab4, text="4")
    ParticleSwarmAlgorithm(tab4,window,ax,canvas)

    tab5 = tk.Frame(tab_control, bg="#008080")
    tab_control.add(tab5, text="5")
    BeesAlgorithm(tab5,window,ax,canvas)

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
    plot_surface(ax, "Функция Химмельблау")
    canvas.draw()
    window.mainloop()
if __name__ == "__main__":
    main()
