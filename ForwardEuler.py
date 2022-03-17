import time

import matplotlib.pyplot as plt
import numpy as np

# global constants here
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
    :param f: function. RHS of differential equation.
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
    :param f: function. RHS of differential equation.
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
        # print(f(U[i], i * k))

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


# Try bad timesteps and compare them to good ones
start = time.perf_counter()
U_1, err_1 = forward_euler_err(func, 3, 1, 100)
U_2, err_2 = forward_euler_err(func, 3, 0.5, 200)
U_3, err_3 = forward_euler_err(func, 3, 0.25, 400)
end = time.perf_counter()

print(end - start)


# Calculate error ratios - Section A.6.1 of the textbook
num = np.abs(U_1[-1] - U_2[-1])
den = np.abs(U_2[-1] - U_3[-1])
error = num / den

print(error)


# Plot time on the X-axis
fig, ax = plt.subplots()  # Create a figure and an axes.
ax.plot(np.arange(0, 1 * len(U_1), 1), U_1, label=str(len(U_1)) + ' iterations')
ax.plot(np.arange(0, 0.5 * len(U_2), 0.5), U_2, label=str(len(U_2)) + ' iterations')
ax.plot(np.arange(0, 0.25 * len(U_3), 0.25), U_3, label=str(len(U_3)) + ' iterations')
ax.set_xlabel('Time')
ax.set_ylabel('Function value')
ax.set_title("Forward Euler")
ax.legend()
plt.show()


# Missing abs()
fig, ax = plt.subplots()  # Create a figure and axes.
ax.plot(np.arange(0, 1 * len(err_1), 1), err_1, label=str(len(U_1)) + ' iterations')
ax.plot(np.arange(0, 0.5 * len(err_2), 0.5), err_2, label=str(len(U_2)) + ' iterations')
ax.plot(np.arange(0, 0.25 * len(err_3), 0.25), err_3, label=str(len(U_3)) + ' iterations')
ax.set_xlabel('Time')
ax.set_ylabel('Error')
ax.set_title("Forward Euler")
ax.legend()
plt.show()
