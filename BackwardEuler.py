import time

import matplotlib.pyplot as plt
import numpy as np

# global constants here
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


def backward_euler(f, Df, eta, k, N, M):
    """
    Implementation of backward Euler. This is a simple
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
        g = lambda u: u - k * f(u, i * k) - U[i]
        Dg = lambda u: 1 - k * Df(u, i * k)
        U[i + 1] = newtons_method(g, Dg, U[i], M)

    return U


def backward_euler_err(f, Df, eta, k, N, M):
    """
    Implementation of backward Euler. This is a simple
    technique used to solve differential equations numerically.
    The function also calculates an error.
    :param f: function. RHS of the differential equation
    :param Df: derivative of f
    :param eta: initial condition
    :param k: time step
    :param N: total number of iterations
    :param M: total number of Newton's method iterations
    :return: list of values
    """
    U = np.zeros(N + 1)
    err = np.zeros(N + 1)
    U[0] = eta

    for i in range(N):
        g = lambda u: u - k * f(u, i * k) - U[i]
        Dg = lambda u: 1 - k * Df(u, i * k)
        w = U[i]
        U[i + 1] = newtons_method(g, Dg, w, M)
        err[i + 1] = eta * np.exp(a * (i + 1) * k)

    return U, err


def backward_euler_system(f, Df, eta, k, N, M):
    """
    Implementation of backward Euler to solve a system
    of differential equations numerically.
    :param f: function. RHS of the differential equation
    :param Df: derivative of f
    :param eta: initial condition
    :param k: time step
    :param N: total number of iterations
    :param M: total number of Newton's method iterations
    :return: list of values
    """
    M = len(eta)
    U = np.zeros(M, N + 1)
    U[:, 0] = eta

    for i in range(N):
        g_i = lambda u: u - k * f(u, i * k) - U[i]
        Dg_i = lambda u: 1 - k * Df(u, i * k)

        for j in range(M):
            w = U[i]
            U[i + 1] = newtons_method(g_i, Dg_i, w, M)

    return U


# The rest of the code is for timing studies, errors, and plots
# using the backward Euler functions.
# Time each of these runs separately. Iterations vs time passed.

start = time.perf_counter()
U_1 = backward_euler(func, Dfunc, 1.0, 0.1, 20, 400)
U_2 = backward_euler(func, Dfunc, 1.0, 0.05, 40, 400)
U_3 = backward_euler(func, Dfunc, 1.0, 0.025, 80, 400)
end = time.perf_counter()

print(end - start)

# Calculate error ratios - Section A.6.1 of the textbook
num = np.abs(U_1[-1] - U_2[-1])
den = np.abs(U_2[-1] - U_3[-1])
error = num / den

print(error)

# Plot time on the X-axis
fig, ax = plt.subplots()
ax.plot(np.arange(0, 1 * len(U_1), 1), U_1, label=str(len(U_1)) + ' iterations')
ax.plot(np.arange(0, 0.5 * len(U_2), 0.5), U_2, label=str(len(U_2)) + ' iterations')
ax.plot(np.arange(0, 0.25 * len(U_3), 0.25), U_3, label=str(len(U_3)) + ' iterations')
ax.set_xlabel('Time')
ax.set_ylabel('Function value')
ax.set_title("Backward Euler")
ax.legend()
plt.show()
