# Задание 6.3
#
# Формат файла:
# n m
# Матрица [n, m], построчно
# b

import numpy as np

from hw06.task1 import semicholesky, read_matrix_file


# Минимизирует по x значение ||Ax - b||**2
def solve_axb(A, b):
    n = A.shape[0]
    m = A.shape[1]
    # В предыдущих дз доказывали, что ||Ax - b|| ** 2 минимизируется при A.T * A * x = A.T * b
    # A.T * A не обязана быть положительно определенной, нас устроит произвольное разложение Холецкого.
    L = semicholesky(np.dot(A.T, A))
    new_b = np.dot(A.T, b)

    # L * y = new_b
    y = np.zeros(m)
    for i in range(m):
        y[i] = new_b[i] - sum(L[i, :i] * y[:i])
        if L[i][i] != 0:
            y[i] /= L[i][i]

    # L.T * x = y
    x = np.zeros(m)
    for i in range(m - 1, -1, -1):
        x[i] = y[i] - sum(L.T[i, i + 1:] * x[i + 1:])
        if L.T[i][i] != 0:
            x[i] /= L.T[i][i]
    return x


def main():
    f = open('input', "r")
    # n = int(input("Enter size of matrix: "))
    n, m = map(int, f.readline().split())
    A = read_matrix_file(n, m, f)
    b = np.fromstring(f.readline(), dtype=float, sep=' ')

    # Линейная независимость <=> не существует x: x.T * A = 0 <=> не существует x: x.T * A * (x.T * A).T = 0 <=>
    # <=> не существует x: x.T * (A * A.T) * x = 0 <=> из-за симметричности A * A.T, A * A.T должна быть положительно
    # определенной. Это и проверим.
    # Проверку можно выключить, если хочется минимизировать систему в более общем случае. Алгоритм при этом работает.
    check_enabled = True
    if check_enabled and not np.all(np.linalg.eigvals(np.dot(A, A.T)) > 0):
        raise ValueError("Rows aren't linearly independent")

    x = solve_axb(A, b)
    print(np.linalg.norm(np.dot(A, x) - b) ** 2)
    print(x)


if __name__ == '__main__':
    main()
