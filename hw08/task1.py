import numpy as np


def fun(x):
    return 0.5 * (x[0] ** 2 + 69 * x[1] ** 2)


def grad(x):
    return np.array([1. * x[0], 69. * x[1]])


def heavy_ball(steps, x_0):
    m = 1.
    M = 69.
    alpha = 4. / ((np.sqrt(M) + np.sqrt(m)) ** 2)
    beta = (np.sqrt(M) - np.sqrt(m)) / (np.sqrt(M) + np.sqrt(m))
    x = x_0.copy()
    delta = np.zeros(2)
    for i in range(steps):
        last_x = x.copy()
        x = last_x - alpha * grad(last_x) + beta * delta
        delta = x - last_x
    print("Heavy ball:\nx = {0}\nf(x) = {1}".format(x, fun(x)))


def nesterov(steps, x_0):
    m = 1.
    M = 69.
    # m != 0, можем пользоваться упрощенной схемой с a_k = sqrt{m/M}
    beta = (np.sqrt(M) - np.sqrt(m)) / (np.sqrt(M) + np.sqrt(m))
    x = x_0.copy()
    y = x_0.copy()
    for i in range(steps):
        last_x = x.copy()
        last_y = y.copy()
        x = last_y - 1. / M * grad(last_y)
        y = x + beta * (x - last_x)
    print("Nesterov:\nx = {0}\ny = {1}\nf(x) = {2}".format(x, y, fun(x)))


def main():
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    x_0 = np.array([1, 1])
    print("Starting from x = {0}".format(x_0))
    steps = int(input("Enter number of iterations: "))
    heavy_ball(steps, x_0)
    nesterov(steps, x_0)


if __name__ == '__main__':
    main()
