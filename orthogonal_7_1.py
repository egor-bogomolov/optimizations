# Задание 7.1
#
# Формат файла:
# n m
# m строк по n чисел -- векторы
# n строк по n чисел -- матрица A

import numpy as np
from cholesky_6_1 import read_matrix_file


def mult(v, A, u):
    return np.dot(np.dot(v.T, A), u)


def main():
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    f = open('input', "r")
    n, m = map(int, f.readline().split())
    v = read_matrix_file(m, n, f)
    A = read_matrix_file(n, n, f)
    u = np.zeros([m, n])
    for i in range(m):
        u[i] = v[i]
        for j in range(i):
            u[i] -= u[j] * sum(v[i] * u[j]) / sum(u[j] * u[j])

    print(u)
    for i in range(m):
        for j in range(i):
            print("%0.3f" % sum(u[i] * u[j]), end=' ')
        print()

    uA = np.zeros([m, n])
    for i in range(m):
        uA[i] = v[i]
        for j in range(i):
            uA[i] -= uA[j] * mult(uA[j], A, v[i]) / mult(uA[j], A, uA[j])

    print(uA)
    for i in range(m):
        for j in range(i):
            print("%0.3f" % mult(uA[j], A, uA[i]), end=' ')
        print()


if __name__ == '__main__':
    main()
