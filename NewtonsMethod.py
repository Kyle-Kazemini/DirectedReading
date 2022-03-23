import numpy as np


def newtons_method(f, Df, w_0, N, epsilon=1e-5):
    """
    Implementation of Newton's method for finding
    the roots of a function.
    :param f: the function we want to find the roots of
    :param Df: derivative of f
    :param w_0: initial guess
    :param epsilon: tolerance
    :param N: number of iterations
    :return: root of f
    """

    w = w_0

    for i in range(N):
        if np.abs(f(w)) < epsilon:
            return w

        w = w - (f(w) / Df(w))

    print("No solution found after max number of iterations.")
    return w


def nd_newtons_method(f, J, w_0, N, epsilon=1e-5):
    """
    Implementation of Newton's method for finding
    the roots of a multi-dimensional function.
    :param f: the function we want to find the roots of
    :param J: Jacobian of f
    :param w_0: initial guess
    :param epsilon: tolerance
    :param N: number of iterations
    :return: root of f
    """

    i = 0
    w = w_0
    val = f(w)
    norm = np.linalg.norm(val, ord=2)

    while i < N and abs(norm) > epsilon:
        delta = np.linalg.solve(J(w), val)
        w = w - delta
        val = f(w)
        norm = np.linalg.norm(val, ord=2)
        i += 1

    if abs(norm) > epsilon:
        i = -1

    return w


def func(X):
    return np.array([X[0] + X[1] - 4, 2 * X[0] - X[1] + 2])


def J(X):
    return np.array([[1, 1], [2, -1]])


print(nd_newtons_method(func, J, (0, 0), 100))
