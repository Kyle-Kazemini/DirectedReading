import time
import numpy as np


a = 1.0


def func(x, t):
    """
    :param x: spatial variable
    :param t: temporal variable
    :param a: global constant
    :return: function
    """
    return a * x


def forward_euler(f, eta, k, N):
    """
    Function to perform forward Euler. This is a simple technique
    to solve differential equations numerically.
    :param f: function. RHS of differential equation
    :param eta: initial value
    :param k: time step
    :param N: number of iterations
    :return:
    """
    U = np.zeros(N + 1)
    U[0] = eta

    for i in range(N):
        U[i + 1] = U[i] + k * func(U[i], i * k)

    return U


def forward_euler_err(f, eta, k, N):
    """
    Function to perform forward Euler. This is a simple technique
    to solve differential equations numerically.
    The function also calculates an error.
    :param f: function. RHS of differential equation
    :param eta: initial value
    :param k: time step
    :param N: number of iterations
    :return:
    """
    U = np.zeros(N + 1)
    err = np.zeros(N + 1)
    U[0] = eta

    for i in range(N):
        U[i + 1] = U[i] + k * f(U[i], i * k)
        err[i + 1] = U[i + 1] - eta * np.exp(a * (i + 1) * k)

    return U, err


def forward_euler_system(f, eta, k, N):
    """
    Function to perform forward Euler on a system. This is a simple technique
    to solve differential equations numerically.
    :param f: function. RHS of differential equation
    :param eta: initial value
    :param k: time step
    :param N: number of iterations
    :return:
    """
    t = 0
    M = len(eta)
    U = np.zeros(M, N + 1)
    U[:, 0] = eta

    for i in range(N):
        f_i = f(U[:, i], t)
        for j in range(M):
            U[j, i + 1] = U[j, i] + k * f_i[j]
        t += k

    return U


def SIR(x, t, gamma, beta):
    return np.array([-beta * x[0] * x[1],
                     beta * x[0] * x[1] - gamma * x[1],
                     gamma * x[1]])
