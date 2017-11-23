# Задание 7.4
#
# Класс для быстрого умножения матриц друг на друга и на векторы слева/справа

import numpy as np


class SparseMatrix:

    @staticmethod
    def compress_row(row):
        compressed = []
        for i in range(len(row)):
            val = row[i]
            if val != 0:
                compressed.append((i, val))
        return compressed

    @staticmethod
    def mult_rows(row1, row2):
        i = 0
        j = 0
        result = 0
        while i < len(row1) and j < len(row2):
            ind1, val1 = row1[i]
            ind2, val2 = row2[j]
            if ind1 == ind2:
                result += val1 * val2
                i += 1
                j += 1
            elif ind1 < ind2:
                i += 1
            else:
                j += 1
        return result

    def __init__(self, matrix):
        self.n = matrix.shape[0]
        self.m = matrix.shape[1]
        self.horizontal = []
        for row in matrix:
            self.horizontal.append(self.compress_row(row))

        self.vertical = []
        for row in matrix.T:
            self.vertical.append(self.compress_row(row))

    def mult(self, matrix):
        if self.m != matrix.n:
            raise ValueError("Matrices can't be multiplied.")
        result = np.zeros([self.n, matrix.m])
        for i in range(self.n):
            for j in range(matrix.m):
                result[i][j] = self.mult_rows(self.horizontal[i], matrix.vertical[j])
        return result

    def mult_vector_right(self, vector):
        if vector.shape[0] != self.m:
            raise ValueError("Matrix and vector can't be multiplied.")
        result = np.zeros(self.n)
        for i in range(self.n):
            for ind, val in self.horizontal[i]:
                result[i] += val * vector[ind]
        return result

    def mult_vector_left(self, vector):
        if vector.shape[0] != self.n:
            raise ValueError("Matrix and vector can't be multiplied.")
        result = np.zeros(self.m)
        for i in range(self.m):
            for ind, val in self.vertical[i]:
                result[i] += val * vector[ind]
        return result
