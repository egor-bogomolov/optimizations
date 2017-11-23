# Задание 7.5
#
# Формат файла:
# n m
# m строк, на i-ой концы i-ого ребра, нумерация с 1
# sigma, n чисел
#
# Пронумеруем ребра числами от 1 до m.
# Запишем требуемое равенство так:
# M * w = sigma
# M -- матрица инцидентности, n строк, m столбцов, M_{i, j} равно 1 если ребро j входит в вершину i, -1 если выходит из
# неё и 0 иначе. w -- вектор весов ребер
# Это все попадает под определение "произвольной вещественной матрицы", можем решать с помощью предыдущих заданий.

import numpy as np

from hw06.task3 import solve_axb


def main():
    precision = 1e-9
    f = open('input', "r")
    n, m = map(int, f.readline().split())
    M = np.zeros([n, m])
    edges = []
    for i in range(m):
        a, b = map(int, f.readline().split())
        edges.append((a, b))
        M[a - 1][i] = -1
        M[b - 1][i] = 1
    sigma = np.fromstring(f.readline(), dtype=float, sep=' ')
    w = solve_axb(M, sigma)
    if np.linalg.norm(np.dot(M, w) - sigma) > precision:
        print("There is no solution")
    else:
        for i in range(m):
            print(edges[i], "w = %0.3f" % w[i])


if __name__ == '__main__':
    main()
