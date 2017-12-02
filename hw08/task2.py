import numpy as np


def fun(x):
    return np.e ** (x[0] + 3. * x[1] - 0.1) + np.e ** (x[0] - 3. * x[1] - 0.1) + np.e ** (-x[0] - 0.1)


# Равен 0 при x[0] = -ln(2) / 2, x[1] = 0. Это и есть искомый минимум.
def grad(x):
    return np.array([np.e ** (x[0] + 3. * x[1] - 0.1) + np.e ** (x[0] - 3. * x[1] - 0.1) - np.e ** (-x[0] - 0.1),
                     3. * np.e ** (x[0] + 3. * x[1] - 0.1) - 3. * np.e ** (x[0] - 3. * x[1] - 0.1)])


def nesterov(steps, x_0, m, M, min_p):
    beta = (np.sqrt(M) - np.sqrt(m)) / (np.sqrt(M) + np.sqrt(m))
    x = x_0.copy()
    y = x_0.copy()
    for i in range(steps):
        last_x = x.copy()
        last_y = y.copy()
        x = last_y - 1. / M * grad(last_y)
        y = x + beta * (x - last_x)
    print("Nesterov:\nx = {0}\ny = {1}\nf(x) = {2}".format(x, y, fun(x)))
    print("error = {0}".format(fun(x) - fun(min_p)))


def main():
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    x_0 = np.array([1, 1])
    print("Starting from x = {0}".format(x_0))
    steps = int(input("Enter number of iterations: "))
    m = 0
    M = 0
    x_max = max(1, abs(x_0[0]))
    y_max = max(1, abs(x_0[1]))
    for i in range(1000):
        coords = np.random.random([4])
        p1 = np.array([coords[0] * 2 * x_max - x_max, coords[1] * 2 * y_max - y_max])
        p2 = np.array([coords[2] * 2 * x_max - x_max, coords[3] * 2 * y_max - y_max])
        t = sum((grad(p1) - grad(p2)) * (p1 - p2)) / (np.linalg.norm(p1 - p2) ** 2)
        M = max(M, t)
        if m == 0:
            m = t
        else:
            m = min(m, t)
    m /= 2
    M *= 2
    min_p = np.array([-np.log(2.) / 2., 0.])
    nesterov(steps, x_0, m, M, min_p)


if __name__ == '__main__':
    main()
