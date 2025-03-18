import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
import numpy as np
import math
import numpy
from abc import ABCMeta, abstractmethod

import numpy

import numpy.random


class Particle (object):
    """
    Класс, описывающий одну частицу
    """
    def __init__ (self, swarm):
        """
        swarm - экземпляр класса Swarm, хранящий параметры алгоритма, список частиц и лучшее значение роя в целом
        position - начальное положение частицы (список)
        """
        # Текущее положение частицы
        self.__currentPosition = self.__getInitPosition (swarm)

        # Лучшее положение частицы
        self.__localBestPosition = self.__currentPosition[:]

        # Лучшее значение целевой функции
        self.__localBestFinalFunc = swarm.getFinalFunc (self.__currentPosition)

        self.__velocity = self.__getInitVelocity (swarm)


    @property
    def position (self):
        return self.__currentPosition


    @property
    def velocity (self):
        return self.__velocity


    def __getInitPosition (self, swarm):
        """
        Возвращает список со случайными координатами для заданного интервала изменений
        """
        return numpy.random.rand (swarm.dimension) * (swarm.maxvalues - swarm.minvalues) + swarm.minvalues


    def __getInitVelocity (self, swarm):
        """
        Сгенерировать начальную случайную скорость
        """
        assert len (swarm.minvalues) == len (self.__currentPosition)
        assert len (swarm.maxvalues) == len (self.__currentPosition)

        minval = -(swarm.maxvalues - swarm.minvalues)
        maxval = (swarm.maxvalues - swarm.minvalues)

        return numpy.random.rand (swarm.dimension) * (maxval - minval) + minval


    def nextIteration (self, swarm):
        # Случайный вектор для коррекции скорости с учетом лучшей позиции данной частицы
        rnd_currentBestPosition = numpy.random.rand (swarm.dimension)

        # Случайный вектор для коррекции скорости с учетом лучшей глобальной позиции всех частиц
        rnd_globalBestPosition = numpy.random.rand (swarm.dimension)

        veloRatio = swarm.localVelocityRatio + swarm.globalVelocityRatio
        commonRatio = (2.0 * swarm.currentVelocityRatio / 
                (numpy.abs (2.0 - veloRatio - numpy.sqrt (veloRatio ** 2 - 4.0 * veloRatio) ) ) )

        # Посчитать новую скорость
        newVelocity_part1 = commonRatio * self.__velocity

        newVelocity_part2 = (commonRatio * 
                swarm.localVelocityRatio * 
                rnd_currentBestPosition * 
                (self.__localBestPosition - self.__currentPosition) )

        newVelocity_part3 = (commonRatio * 
                swarm.globalVelocityRatio * 
                rnd_globalBestPosition * 
                (swarm.globalBestPosition - self.__currentPosition) )
        
        self.__velocity = newVelocity_part1 + newVelocity_part2 + newVelocity_part3

        # Обновить позицию частицы
        self.__currentPosition += self.__velocity

        finalFunc = swarm.getFinalFunc (self.__currentPosition)


class Swarm (object):
    """
    Базовый класс для роя частиц. Его надо переопределять для конкретной целевой функции
    """
    __metaclass__ = ABCMeta

    def __init__ (self, 
            swarmsize, 
            minvalues, 
            maxvalues, 
            currentVelocityRatio,
            localVelocityRatio, 
            globalVelocityRatio):
        """
        swarmsize - размер роя (количество частиц)
        minvalues - список, задающий минимальные значения для каждой координаты частицы
        maxvalues - список, задающий максимальные значения для каждой координаты частицы
        currentVelocityRatio - общий масштабирующий коэффициент для скорости
        localVelocityRatio - коэффициент, задающий влияние лучшей точки, найденной частицей на будущую скорость
        globalVelocityRatio - коэффициент, задающий влияние лучшей точки, найденной всеми частицами на будущую скорость
        """
        self.__swarmsize = swarmsize

        assert len (minvalues) == len (maxvalues)
        assert (localVelocityRatio + globalVelocityRatio) > 4

        self.__minvalues = numpy.array (minvalues[:])
        self.__maxvalues = numpy.array (maxvalues[:])

        self.__currentVelocityRatio = currentVelocityRatio
        self.__localVelocityRatio = localVelocityRatio
        self.__globalVelocityRatio = globalVelocityRatio

        self.__globalBestFinalFunc = None
        self.__globalBestPosition = None

        self.__swarm = self.__createSwarm ()


    def __getitem__ (self, index):
        """
        Возвращает частицу с заданным номером
        """
        return self.__swarm[index]


    def __createSwarm (self):
        """
        Создать рой из частиц со случайными координатами
        """
        return [Particle (self) for _ in range (self.__swarmsize) ]



    def nextIteration (self):
        """
        Выполнить следующую итерацию алгоритма
        """
        for particle in self.__swarm:
            particle.nextIteration (self)


    @property
    def minvalues (self):
        return self.__minvalues


    @property
    def maxvalues (self):
        return self.__maxvalues


    @property
    def currentVelocityRatio (self):
        return self.__currentVelocityRatio


    @property
    def localVelocityRatio (self):
        return self.__localVelocityRatio


    @property
    def globalVelocityRatio (self):
        return self.__globalVelocityRatio


    @property
    def globalBestPosition (self):
        return self.__globalBestPosition


    @property
    def globalBestFinalFunc (self):
        return self.__globalBestFinalFunc


    def getFinalFunc (self, position):
        assert len (position) == len (self.minvalues)

        finalFunc = self._finalFunc (position)

        if (self.__globalBestFinalFunc == None or
                finalFunc < self.__globalBestFinalFunc):
            self.__globalBestFinalFunc = finalFunc
            self.__globalBestPosition = position[:]


    @abstractmethod
    def _finalFunc (self, position):
        pass

    
    @property
    def dimension (self):
        """
        Возвращает текущую размерность задачи
        """
        return len (self.minvalues)


    def _getPenalty (self, position, ratio):
        """
        Рассчитать штрафную функцию
        position - координаты, для которых рассчитывается штраф
        ratio - вес штрафа
        """
        penalty1 = sum ([ratio * abs (coord - minval)
            for coord, minval in zip (position, self.minvalues) 
            if coord < minval ] )

        penalty2 = sum ([ratio * abs (coord - maxval)
            for coord, maxval in zip (position, self.maxvalues) 
            if coord > maxval ] )

        return penalty1 + penalty2



class Swarm_Rastrigin (Swarm):
    def __init__ (self,
            swarmsize,
            minvalues,
            maxvalues,
            currentVelocityRatio,
            localVelocityRatio,
            globalVelocityRatio):
       Swarm.__init__ (self,
            swarmsize,
            minvalues,
            maxvalues,
            currentVelocityRatio,
            localVelocityRatio,
            globalVelocityRatio)


    def _finalFunc (self, position):
        function = 10.0 * len (self.minvalues) + sum (position * position - 10.0 * numpy.cos (2 * numpy.pi * position) )
        penalty = self._getPenalty (position, 10000.0)

        return function + penalty



# инерция
# альфа
# бетта

def printResult (swarm, iteration):
    template = u""" Лучшие координаты: {bestpos}\n Лучший результат: {finalfunc}\n\n"""

    result = template.format (iter = iteration,
            bestpos = swarm.globalBestPosition,
            finalfunc = swarm.globalBestFinalFunc)

    return result


def ParticleSwarmAlgorithm(frame,root,ax,canvas):

        def rastrigin(*X):
            A = 10
            return A + sum([(x ** 2 - A * np.cos(2 * math.pi * x)) for x in X])

        def run_optimization():
            # Генерация сетки для графика целевой функции
            X = np.linspace(-4, 4, 200)
            Y = np.linspace(-4, 4, 200)

            X, Y = np.meshgrid(X, Y)
            Z = rastrigin(X, Y)


            iterCount = iteration.get()
            dimension = 3
            swarmsize = particle.get()
            minvalues = numpy.array ([-5.12] * dimension)
            maxvalues = numpy.array ([5.12] * dimension)

            currentVelocityRatio = inertia.get()
            localVelocityRatio = alpha.get()
            globalVelocityRatio = beta.get()

            swarm = Swarm_Rastrigin(swarmsize,
                                    minvalues,
                                    maxvalues,
                                    currentVelocityRatio,
                                    localVelocityRatio,
                                    globalVelocityRatio
                                    )


            #для записи результатов
            results = []
            results_text.config(state=tk.NORMAL)
            results_text.delete(1.0, tk.END)

            for n in range(iterCount):
                ax.cla()
                # Построение поверхности графика целевой функции
                ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5)
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.set_xticks(np.arange(-4, 4 + 1, 2))
                ax.set_yticks(np.arange(-4, 4 + 1, 2))
                ax.set_title("Алгоритм Роя Частиц")

                ax.scatter(swarm[0].position[0], swarm[0].position[1], swarm[0].position[2], color='red',
                               s=10)

                results_text.insert(tk.END,
                                    f"Итерация {n}\n")

                results_text.insert(tk.END,
                                    f"Позиция {swarm[0].position}\n")
                results_text.insert(tk.END,
                                    f"Скорость {swarm[0].velocity}\n")



                results_text.insert(tk.END,printResult(swarm, n))
                swarm.nextIteration()
                results_text.yview_moveto(1)
                canvas.draw()
                root.update()


        param_frame2 = frame
        bg_color = "#3d6466"
        param_frame2.configure(bg=bg_color)

        # Параметры задачи
        ttk.Label(param_frame2, text="Инициализация значений", font=("Helvetica", 12,"bold")).grid(row=0, column=0, pady=15)
        ttk.Label(param_frame2, text="Частиц", font=("Helvetica", 10)).grid(row=2, column=0)
        ttk.Label(param_frame2, text="Итераций", font=("Helvetica", 10)).grid(row=1, column=0)
        ttk.Label(param_frame2, text="Альфа", font=("Helvetica", 10)).grid(row=3, column=0)
        ttk.Label(param_frame2, text="Бета", font=("Helvetica", 10)).grid(row=4, column=0)
        ttk.Label(param_frame2, text="Инерция", font=("Helvetica", 10)).grid(row=5, column=0)

        #частиц
        particle = tk.IntVar(value=2000)
        iteration = tk.IntVar(value=100)
        alpha=tk.IntVar(value=2)
        beta=tk.IntVar(value=5)
        inertia=tk.DoubleVar(value=0.5)

        particle_entry = ttk.Entry(param_frame2, textvariable=particle)
        iteration_entry = ttk.Entry(param_frame2, textvariable=iteration)
        alpha_entry = ttk.Entry(param_frame2, textvariable=alpha)
        beta_entry = ttk.Entry(param_frame2, textvariable=beta)
        inertia_entry = ttk.Entry(param_frame2, textvariable=inertia)

        particle_entry.grid(row=2, column=1)
        iteration_entry.grid(row=1, column=1)
        alpha_entry.grid(row=3, column=1)
        beta_entry.grid(row=4, column=1)
        inertia_entry.grid(row=5, column=1)


        separator = ttk.Separator(param_frame2, orient="horizontal")  # Горизонтальная полоса разделения
        separator.grid(row=7, column=0, columnspan=2, sticky="ew", pady=10)

        # Параметры функции
        ttk.Label(param_frame2, text="Функция и отображение ее графика", font=("Helvetica", 12, "bold")).grid(row=9, column=0, pady=10)
        ttk.Label(param_frame2, text="Выберите функцию", font=("Helvetica", 10)).grid(row=10, column=0)
        function_choices = ["Функция Растригина"]
        function_var = tk.StringVar(value=function_choices[0])
        function_menu = ttk.Combobox(param_frame2, textvariable=function_var, values=function_choices, width=22)
        function_menu.grid(row=10, column=1, pady=5)

        separator = ttk.Separator(param_frame2, orient="horizontal")  # Горизонтальная полоса разделения
        separator.grid(row=18, column=0, columnspan=2, sticky="ew", pady=10)

        # Создание кнопки Выполнить
        button_style = ttk.Style()
        button_style.configure("My.TButton", font=("Helvetica", 14))

        # Создание кнопки Выполнить
        apply_settings_button = ttk.Button(param_frame2, text="Выполнить",command=run_optimization, style="My.TButton")
        apply_settings_button.grid(row=21, column=1, padx=10, pady=10)

        ttk.Label(param_frame2, text="Выполнение и результаты", font=("Helvetica", 12, "bold")).grid(row=20, column=0, pady=10)
        results_text = scrolledtext.ScrolledText(param_frame2, wrap=tk.WORD, height=18, width=40, padx=2, state=tk.DISABLED)
        results_text.grid(row=21, column=0, padx=10)
        

