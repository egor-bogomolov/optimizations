# Задание 7.2
#
# Раскроем скобки, получим следующее выражение под интегралом:
# sum_{i, j} a_i * a_j * x^{i+j}, если считать a_0 = 1
# Возьмем интеграл:
# sum_{i, j} a_i * a_j * b_{i+j}, b_{i+j} = 0, если i+j нечетное, иначе 2/(i+j+1)
# Возьмем производные по a_i и приравняем их 0, получим уравнения для 1 <= i <= n:
# sum_{j=0}^{n} a_j * b_{i+j} = 0
# Или же:
# sum_{j=1}^{n} a_j * b_{i+j} = -b_i
# Получаем следующую систему:
# B * a + c = 0,
# Где:
# B_{i, j} = b_{i+j}, нумерация с 1
# c = (b_1, ..., b_n)
# Матрица B положительно определенная, потому что на всех позициях с нечетным i+j стоят 0. Решение точно найдется.

import numpy as np

from hw06.task3 import solve_axb


def get_b(x):
    if (x % 2) != 0:
        return 0
    else:
        return 2. / (x + 1)


def main():
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    n = int(input())
    B = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            B[i][j] = get_b(i + j + 2)
    c = np.zeros(n)
    for i in range(n):
        c[i] = -get_b(i + 1)
    a = solve_axb(B, c)
    print("coefficients =", a)
    res = 0
    for i in range(n + 1):
        for j in range(n + 1):
            val_i = 1
            val_j = 1
            if i != 0:
                val_i = a[i - 1]
            if j != 0:
                val_j = a[j - 1]
            res += val_i * val_j * get_b(i + j)
    print("norm = %0.3f" % res)


if __name__ == '__main__':
    main()
