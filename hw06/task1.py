# Задание 6.1
#
# Формат файла:
# n
# Матрица [n, n], построчно

import numpy as np


# Рекурсивная реализация разложения Холецкого для положительно определенных матриц.
# Для полуопределенных используйте semicholesky.
def cholesky(matrix):
    if not np.allclose(matrix, matrix.T, atol=1e-9):
        raise ValueError("Matrix isn't symmetric")

    n = matrix.shape[0]
    result = np.zeros([n, n])
    for i in range(n):
        if matrix[i][i] < 0:
            raise ValueError("Matrix isn't positive definite")
        sqrt = np.sqrt(matrix[i][i])
        result[i][i] = sqrt
        line = np.array([matrix[i + 1:, i]])
        result[i + 1:, i] = line / sqrt
        matrix[i + 1:, i + 1:] -= np.dot(line.T, line) / matrix[i][i]

    return result


# Нерекурсивная реализация разложениея Холецкого. Допускает положительно полуопределенные матрицы и возвращает для
# произвольное корректное разложение.
def semicholesky(matrix):
    if not (matrix.T == matrix).all():
        raise ValueError("Matrix isn't symmetric")

    n = matrix.shape[0]
    result = np.zeros([n, n])
    for i in range(n):
        for j in range(i):
            if result[j][j] != 0:
                result[i][j] = (matrix[i][j] - sum(result[i, :j] * result[j, :j])) / result[j][j]
        value = matrix[i][i] - sum(result[i, :i] ** 2)
        if value < 0:
            raise ValueError("Matrix isn't positive semi-definite")
        result[i][i] = np.sqrt(value)

    return result


def read_matrix_file(n, m, f):
    matrix = np.zeros([n, m])
    for i in range(n):
        matrix[i] = np.fromstring(f.readline(), dtype=float, sep=' ')
    return matrix


def read_matrix_input(n, m):
    matrix = np.zeros([n, m])
    for i in range(n):
        matrix[i] = np.fromstring(input("Enter row " + i.__str__() + ": "), dtype=float, sep=' ')
    return matrix


def main():
    f = open('input', "r")
    # n = int(input("Enter size of matrix: "))
    n = int(f.readline())
    matrix = read_matrix_file(n, n, f)
    result = cholesky(matrix)
    print(result)
    print(np.dot(result, result.T))
    np.linalg.cholesky(matrix)


if __name__ == '__main__':
    main()
