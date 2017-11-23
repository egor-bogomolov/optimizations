# Задание 6.2
#
# Формат файла:
# n m
# x_i, y_i, по паре на строке, m строк

import numpy as np
import matplotlib.pyplot as plt
from Axb_6_3 import solve_axb


def main():
    f = open('input', "r")
    # n = int(input("Enter size of matrix: "))
    n, m = map(int, f.readline().split())
    x = np.zeros(m)
    y = np.zeros(m)
    for i in range(m):
        x[i], y[i] = map(float, f.readline().split())

    # Хотим минимизировать sum((P(x_i) - y_i) ** 2). Пусть P(x) = a_n * x^n + ... + a_0. Сумма квадратов это квадрат
    # следующей нормы: ||Pa - y|| ** 2, где i-я строка P это 1, x_i, ..., x_i^n, всего строк m.
    # Получили задачу минимизации ||Ax - b||, которую решали раньше.
    P = np.zeros([m, n + 1])
    for i in range(m):
        P[i][0] = 1
        for j in range(n):
            P[i][j + 1] = P[i][j] * x[i]
    a = solve_axb(P, y)
    print(a)
    print(np.linalg.norm(np.dot(P, a) - y) ** 2)
    poly = lambda z: sum(a[k] * (z ** k) for k in range(n + 1))

    min_x = min(x) - 1
    max_x = max(x) + 1
    number_of_points = 1000
    xvals = np.arange(min_x, max_x, (max_x - min_x) / number_of_points)
    yvals = poly(xvals)
    min_y = min(min(y), min(yvals))
    max_y = max(max(y), max(yvals))
    number_of_ticks = 14

    fig = plt.figure()
    ax = fig.gca()
    ax.set_xticks(np.arange(min_x - 1, max_x + 1, int((max_x - min_x) / number_of_ticks) + 1))
    ax.set_yticks(np.arange(int(min_y) - 1, int(max_y) + 1, int((max_y - min_y) / number_of_ticks) + 1))

    ticklines = ax.get_xticklines() + ax.get_yticklines()
    gridlines = ax.get_xgridlines() + ax.get_ygridlines()

    for line in ticklines:
        line.set_linewidth(3)

    for line in gridlines:
        line.set_linestyle('--')

    plt.plot(xvals, yvals)
    plt.title('Polynomial fit')
    plt.grid()
    plt.scatter(x, y, c='red')
    plt.show()


if __name__ == '__main__':
    main()
