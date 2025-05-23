from functions import *
from helper import *

def simplex_method():
    c = vect()
    a = extract_and_modify2()
    d = [0, 0, 0]


    c1 = c.copy()


    def maxVal(a, c):
        """Поиск индексов минимального элемента в последней строке"""
        up_f = max(c[1:])
        max_index = c[1:].index(up_f) + 1

        max_val = []
        for i in range(len(a)):
            if a[i][max_index] != 0:
                ratio = a[i][0] / a[i][max_index]
                max_val.append((ratio, i))

        if max_val:
            non_negative_values = [(ratio, idx) for ratio, idx in max_val if ratio >= 0]
            if non_negative_values:
                min_res, max_res_index = min(non_negative_values, key=lambda x: x[0])
            else:
                min_res, max_res_index = None, None
        else:
            min_res, max_res_index = None, None 
        return [max_res_index, max_index]

    def ch_ab(a, m, c1):
        """Изменение матрицы и вектора c1 (нормализация строки и обновление)"""
        id1, id2 = m
        el = a[id1][id2]

        if el == 0:
            return a, c1

        timeC=(-c1[id2])
        for i in range(len(a[0])):
            a[id1][i] /= el
            c1[i] += a[id1][i] * timeC

        for i in range(len(a)):
            if i != id1:
                time = -a[i][id2]
                for j in range(len(a[0])):
                    a[i][j] += a[id1][j] * time

        return a, c1

    iterations = 0
    max_iterations = 10

    tt=[7, 8, 6]
    h=[]
    while True:
        m = maxVal(a, c1)
        if m[0] is None or m[1] is None or iterations >= max_iterations:
            print("Программа завершена из-за зацикливания или превышения лимита итераций.")
            break
        
        id1, id2 = m
        tt[id1] = id2

        a, c1 = ch_ab(a, m, c1)

        d = [row[0] for row in a]
        if all(x <= 0 for x in c1):
            break

        iterations += 1

        result = [0.0] * 9
        for index, value in zip(tt, d):
            result[index-1] = value
        h.append((round(result[0], 2), round(result[1], 2), round(func1(result[0:2]), 6)))

        f=func1(result[0:2])

    result = [0.0] * 9
    for index, value in zip(tt, d):
        result[index-1] = value
    result_rounded = [round(value, 2) for value in result]

    f=func1(result[0:2])
    h.append((round(result[0],2), round(result[1], 2), round(func1(result[0:2]), 6)))
    return result_rounded, round(f, 6), h

# from functions import *
# from helper import *

# def simplex_method():
#     c = vect()
#     a = extract_and_modify2()
#     d = [0, 0, 0]


#     c1 = c.copy()
#     def lastf(a, d):
#         c_sum = []
#         for j in range(len(a[0])):
#             sum = 0
#             for i in range(len(a)):
#                 sum += d[i] * a[i][j]
#             c_sum.append(sum - c1[j])
#         return c_sum
#     def ch_d(d, m): # изменяем d
#         index1, index2 = m
#         d[index1] = c1[index2]
#         return d
#     def maxVal(a, c):
#         """Поиск индексов минимального элемента в последней строке"""
#         up_f = max(c[1:])
#         max_index = c[1:].index(up_f) + 1

#         max_val = []
#         for i in range(len(a)):
#             if a[i][max_index] != 0:
#                 ratio = a[i][0] / a[i][max_index]
#                 max_val.append((ratio, i))
#         max_res = min(max_val)
#         max_res_index = max_val.index(max_res)
#         return [max_res_index, max_index]

#     def ch_ab(a, m):
#         """Изменение матрицы и вектора c1 (нормализация строки и обновление)"""
#         id1, id2 = m
#         el = a[id1][id2]

#         for i in range(len(a[0])):
#             a[id1][i] /= el

#         for i in range(len(a)):
#             if i != id1:
#                 time = -a[i][id2]
#                 for j in range(len(a[0])):
#                     a[i][j] += a[id1][j] * time

#         return a
#     def x_res(a):
#         arr = []
#         for j in range(len(a[0])):
#             sum0, sum1= 0, 0
#             for i in range(len(a)):
#                 if a[i][j] == 1: sum1 += 1
#                 if a[i][j] == 0: sum0 += 1
#             if int(sum0+sum1) == len(a): arr.append(j)
        
#         return arr
#     while True:
#         c = lastf(a,d)
#         m = maxVal(a, c)
#         print(c)
#         print(m)
#         d = ch_d(d, m)
#         a = ch_ab(a, m)
#         if all(x <= 0 for x in c):
#             break

#     b_vector = [row[0] for row in a]
#     x = x_res(a)

#     max_index = max(x)
#     result=[0]*(max_index+1)
#     for i in range(len(b_vector)):
#         result[x[i]] = b_vector[i]
#     result = result[1:]

#     print("Оптимизированная матрица a:", a)
#     print("Значения вектора d(базис):", d)
#     print("Значения вектора c:", c)
#     print("Значения функции f:", c[0])
#     print("Значения вектора b:", b_vector)
#     print("Вектор x:", result)

# simplex_method()