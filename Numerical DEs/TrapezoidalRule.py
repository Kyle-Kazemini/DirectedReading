import numpy as np
from NewtonsMethod import *

a = 1


def func(x, t):
    """
    :param x: spatial variable
    :param t: temporal variable
    :return: function
    """
    return a * x


def Dfunc(x, t):
    """
    :param x: spatial variable
    :param t: temporal variable
    :return: function
    """
    return a


def nd_func(x, t):
    return np.array(a * x[0], a * x[1])


def nd_Dfunc(x, t):
    return a * np.identity(2)


def trapezoidal_rule(f, Df, eta, k, N, M):
    """
    Implementation of Trapezoidal Rule. This is a simple
    technique used to solve differential equations numerically.
    :param f: function. RHS of the differential equation
    :param Df: derivative of f
    :param eta: initial condition
    :param k: time step
    :param N: total number of iterations
    :param M: total number of Newton's method iterations
    :return: list of values
    """
    U = np.zeros(N + 1)
    U[0] = eta

    for i in range(N):
        g = lambda u: u - U[i] - (k / 2) * (f(u, i * k) + f(U[i], i * k))
        Dg = lambda u: 1 - (k / 2) * Df(u, i * k)
        U[i + 1] = newtons_method(g, Dg, U[i], M)

    return U


def trapezoidal_rule_err(f, Df, eta, k, N, M):
    """
    Implementation of Trapezoidal Rule. This is a simple
    technique used to solve differential equations numerically.
    The function also calculates an error.
    :param f: function. RHS of the differential equation
    :param Df: derivative of f
    :param eta: initial condition
    :param k: time step
    :param N: total number of iterations
    :param M: total number of Newton's method iterations
    :return: list of values and list of errors
    """
    U = np.zeros(N + 1)
    err = np.zeros(N + 1)
    U[0] = eta

    for i in range(N):
        g = lambda u: u - U[i] - (k / 2) * (f(u, i * k) + f(U[i], i * k))
        Dg = lambda u: 1 - (k / 2) * Df(u, i * k)
        U[i + 1] = newtons_method(g, Dg, U[i], M)
        err[i + 1] = eta * np.exp(a * (i + 1) * k)

    return U, err


def trapezoidal_rule_system(f, Df, eta, k, N, M):
    """
    Implementation of Trapezoidal Rule to solve a system
    of differential equations numerically.
    :param f: function. RHS of the differential equation
    :param Df: derivative of f
    :param eta: initial condition
    :param k: time step
    :param N: total number of iterations
    :param M: Newton's method steps
    :return: list of values
    """
    U = np.zeros([len(eta), N + 1])
    U[:, 0] = eta

    for i in range(N):
        def g(u): return u - U[:, i] - (k / 2) * (f(u, i * k) + f(U[:, i], i * k))
        # print(g(U[:, 0]))
        print(U[:, i])
        # print(f(U[:, i], 0))

        def Dg(u): return 1 - (k / 2) * Df(u, i * k)

        U[:, i + 1] = nd_newtons_method(g, Dg, U[:, i], M)

    return U
