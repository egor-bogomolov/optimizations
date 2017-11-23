# Задание 7.3
#
# Формат файла:
# n m
# Матрица [n, m], построчно
# b

import numpy as np
from cholesky_6_1 import read_matrix_file, read_matrix_input
from orthogonal_7_1 import mult


def solve_symmetric(A, b, precision=1e-9):
    n = A.shape[0]
    m = A.shape[1]

    grad = np.zeros([n, n])
    d = np.zeros([n, n])
    x = np.zeros(n)

    for i in range(n):
        grad[i] = np.dot(A, x) - b
        d[i] = grad[i]
        if np.allclose(grad[i], np.zeros(n), atol=precision):
            break
        if i != 0:
            d[i] += sum(grad[i] * grad[i]) / sum(grad[i - 1] * grad[i - 1]) * d[i - 1]
        a = sum(d[i] * grad[i]) / mult(d[i], A, d[i])
        x = x - a * d[i]
    return x


def positive_definite(A, b, precision=1e-9):
    print("Positive definite")
    x = solve_symmetric(A, b, precision)
    print("x = ", x)
    print("Ax = ", np.dot(A, x))


# Уже решали раньше задачу по минимизации ||Ax-b||. Найдем минимизирующий x и если равенство выполнено с некоторой
# точностью, то он и будет решением. Для этого нужно рассмотреть решение A.T * A * x = A.T * b.
def any_matrix(A, b, precision=1e-9):
    print("Any matrix")
    new_A = np.dot(A.T, A)
    new_b = np.dot(A.T, b)
    x = solve_symmetric(new_A, new_b, precision)
    if np.linalg.norm(np.dot(A, x) - b) > precision:
        print("There is no solution.")
    else:
        print("x = ", x)
        print("Ax = ", np.dot(A, x))


def check_matrix(A, precision=1e-9):
    n = A.shape[0]
    m = A.shape[1]
    return n == m and np.allclose(A, A.T, atol=precision) and np.all(np.linalg.eigvals(A) > 0)


def main():
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    f = open('input', "r")
    n, m = map(int, f.readline().split())
    A = read_matrix_file(n, m, f)
    b = np.fromstring(f.readline(), dtype=float, sep=' ')
    if check_matrix(A):
        positive_definite(A, b)
    else:
        print("Matrix isn't symmetric or positive definite.")
    any_matrix(A, b)


if __name__ == '__main__':
    main()
