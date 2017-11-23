# Задание 5

import numpy as np


def print_info(iteration, point, min_p, func):
    print("iteration", iteration)
    print("point", point[0], point[1])
    print("distance to min point", np.linalg.norm(point - min_p))
    print("value", func(point))
    print("error", func(point) - func(min_p))
    print("")


def grad_descent_constant_step(precision, func, gradient, p, step, min_p):
    point = np.copy(p)
    iteration = 0
    opt = func(min_p)
    print_info(iteration, point, min_p, func)
    while func(point) - opt >= precision:
        point -= gradient(point) * step
        iteration += 1
        print_info(iteration, point, min_p, func)


def grad_descent_min_on_line(precision, func, gradient, p, min_p, min_on_line):
    point = np.copy(p)
    iteration = 0
    opt = func(min_p)
    print_info(iteration, point, min_p, func)
    while func(point) - opt >= precision:
        grad = gradient(point)
        point = min_on_line(grad, point, precision)
        iteration += 1
        print_info(iteration, point, min_p, func)


def grad_descent_backtracking_line_search(precision, func, gradient, p, min_p):
    point = np.copy(p)
    gamma = 0.25
    beta = 0.5
    iteration = 0
    inner_iterations = 0
    opt = func(min_p)
    print_info(iteration, point, min_p, func)
    while func(point) - opt >= precision:
        grad = gradient(point)
        func_in_point = func(point)
        step = 1.
        while func(point - step * grad) > func_in_point - gamma * step * (np.linalg.norm(grad) ** 2):
            step *= beta
            inner_iterations += 1
        point -= gradient(point) * step
        iteration += 1
        print("inner iterations", inner_iterations)
        print_info(iteration, point, min_p, func)


def function_1(x):
    return 0.5 * (x[0] ** 2 + 69 * x[1] ** 2)


def gradient_1(x):
    return np.array([1. * x[0], 69. * x[1]])


def min_on_line_1(grad, point, precision):
    # line: y = point[1] + (x - point[0]) * grad[1] / grad[0] = kx + c
    # k = grad[1] / grad[0]
    # c = point[1] - point[0] * grad[1] / grad[0]
    # f = 1 / 2 * (x ** 2 + 69 * y ** 2)
    # f = 1 / 2 * (x ** 2 + 69 * (k * x + c) ** 2)
    # f = 1 / 2 * ((1 + 69 * k ** 2) * x ** 2 + 2 * 69 * kcx + 69 * c ** 2)
    # x_min = - 69kc / (1 + 69k^2)
    # x_min = - 69 * grad[1] * (point[1] * grad[0] - point[0] * grad[1]) / (grad[0]**2 + 69 * grad[1] ** 2)
    # y_min = c / (1 + 69 * k ** 2)
    # y_min = grad[0] * (point[1] * grad[0] - point[0] * grad[1]) / (grad[0]**2 + 69 * grad[1] ** 2)
    x_min = -69 * grad[1] * (point[1] * grad[0] - point[0] * grad[1]) / (grad[0] ** 2 + 69 * grad[1] ** 2)
    y_min = grad[0] * (point[1] * grad[0] - point[0] * grad[1]) / (grad[0]**2 + 69 * grad[1] ** 2)
    return np.array([x_min, y_min])


def function_2(x):
    return np.e ** (x[0] + 3. * x[1] - 0.1) + np.e ** (x[0] - 3. * x[1] - 0.1) + np.e ** (-x[0] - 0.1)


# Равен 0 при x[0] = -ln(2) / 2, x[1] = 0. Это и есть искомый минимум.
def gradient_2(x):
    return np.array([np.e ** (x[0] + 3. * x[1] - 0.1) + np.e ** (x[0] - 3. * x[1] - 0.1) - np.e ** (-x[0] - 0.1),
                     3. * np.e ** (x[0] + 3. * x[1] - 0.1) - 3. * np.e ** (x[0] - 3. * x[1] - 0.1)])


def min_on_line_2(grad, point, precision):
    # line: y = point[1] + (x - point[0]) * grad[1] / grad[0] = kx + c
    # k = grad[1] / grad[0]
    # c = point[1] - point[0] * grad[1] / grad[0]
    # Будем искать минимум тернарным поиском, аналитичекси не выходит, а выпуклость функции позволяет.
    # Чтобы избежать недоразумений при grad[0] = 0, немного повернем прямую.
    if grad[0] == 0:
        grad[0] = 1e-9

    k = grad[1] / grad[0]
    c = point[1] - point[0] * grad[1] / grad[0]
    x_left = -max(2 * abs(max(point)), 10)
    x_right = max(2 * abs(max(point)), 10)
    while x_right - x_left > precision:
        m_left = x_left + (x_right - x_left) / 3
        m_right = x_left + (x_right - x_left) * 2 / 3
        if function_2(np.array([m_left, m_left * k + c])) > function_2(np.array([m_right, m_right * k + c])):
            x_left = m_left
        else:
            x_right = m_right

    x_min = (x_left + x_right) / 2
    y_min = k * x_min + c
    return np.array([x_min, y_min])


def show_results_1(start_point_1, precision):
    m_1 = 1.
    M_1 = 69.
    alpha_1 = 2 / (m_1 + M_1)
    min_point_1 = np.array([0., 0.])
    grad_descent_constant_step(precision, function_1, gradient_1, start_point_1, alpha_1, min_point_1)
    grad_descent_min_on_line(precision, function_1, gradient_1, start_point_1, min_point_1, min_on_line_1)
    grad_descent_backtracking_line_search(precision, function_1, gradient_1, start_point_1, min_point_1)


def show_results_2(start_point_2, precision):
    # Чтобы вычислить допустимый шаг, прикинем константу Липшица для градиента взяв ее максимумом из констант,
    # при которых выполнится неравенство из её определения.
    # Возьмем шаг как обратное к полученному результату
    L = 0.
    x_max = max(1, abs(start_point_2[0]))
    y_max = max(1, abs(start_point_2[1]))
    for i in range(1000):
        coords = np.random.random([4])
        p1 = np.array([coords[0] * 2 * x_max - x_max, coords[1] * 2 * y_max - y_max])
        p2 = np.array([coords[2] * 2 * x_max - x_max, coords[3] * 2 * y_max - y_max])
        t = sum((gradient_2(p1) - gradient_2(p2)) * (p1 - p2)) / (np.linalg.norm(p1 - p2) ** 2)
        L = max(L, t)

    print("Approximate L", L)
    alpha_2 = 1. / L
    min_point_2 = np.array([-np.log(2.) / 2., 0.])
    grad_descent_constant_step(eps, function_2, gradient_2, start_point_2, alpha_2, min_point_2)
    grad_descent_min_on_line(precision, function_2, gradient_2, start_point_2, min_point_2, min_on_line_2)
    grad_descent_backtracking_line_search(eps, function_2, gradient_2, start_point_2, min_point_2)


print(np.array([1, 2]) * np.array([4, 5]))
(x_0, y_0) = map(float, input("Enter x- and y-coordinate of starting point:\n").split())
start_point = np.array([x_0, y_0])
eps = float(input("Enter precision:\n"))
# Закомментируйте лишние вызовы, чтобы посмотреть на результаты одного конкретного запуска.
show_results_1(start_point, eps)
show_results_2(start_point, eps)
