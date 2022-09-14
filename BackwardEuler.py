import time
import numpy as np
from NewtonsMethod import *
import matplotlib.pyplot as plt

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
    :return: list of values and list of errors
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
        def g(u): return u - k * f(u, i * k) - U[i]
        def Dg(u): return 1 - k * Df(u, i * k)

        U[i + 1] = nd_newtons_method(g, Dg, U[i], M)

    return U


# The rest of the code is for timing studies, errors, and plots
# using the backward Euler functions.

times = np.empty(4)

start = time.perf_counter()
U_1, err_1 = backward_euler_err(func, Dfunc, 3, 0.1, 100, 10000)
end = time.perf_counter()
times[0] = (end - start)

start = time.perf_counter()
U_2, err_2 = backward_euler_err(func, Dfunc, 3, 0.05, 200, 10000)
end = time.perf_counter()
times[1] = (end - start)

start = time.perf_counter()
U_3, err_3 = backward_euler_err(func, Dfunc, 3, 0.025, 400, 10000)
end = time.perf_counter()
times[2] = (end - start)

start = time.perf_counter()
U_4, err_4 = backward_euler_err(func, Dfunc, 3, 0.0125, 800, 10000)
end = time.perf_counter()
times[3] = (end - start)

print(times)
iters = np.array([1000, 2000, 4000, 8000])
plt.title("Timing")
plt.xlabel("Iterations")
plt.ylabel("Time Elapsed")
plt.plot(iters, times)
plt.show()

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
