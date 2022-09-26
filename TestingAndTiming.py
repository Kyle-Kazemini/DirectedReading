import time
import matplotlib.pyplot as plt
from BackwardEuler import *

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


print(nd_newtons_method(func, J, (0, 0), 100))


# The rest of the code is for timing studies, errors, and plots
# using the forward Euler functions.

times = np.empty(4)

start = time.time()
U_1, err_1 = forward_euler_err(func, 3, 0.1, 100)
end = time.time()
times[0] = (end - start)

start = time.perf_counter()
U_2, err_2 = forward_euler_err(func, 3, 0.05, 200)
end = time.perf_counter()
times[1] = (end - start)

start = time.perf_counter()
U_3, err_3 = forward_euler_err(func, 3, 0.025, 400)
end = time.perf_counter()
times[2] = (end - start)

start = time.perf_counter()
U_4, err_4 = forward_euler_err(func, 3, 0.0125, 800)
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
# fig, ax = plt.subplots()  # Create a figure and axes.
# ax.plot(np.arange(0, 1 * len(err_1), 1), err_1, label=str(len(U_1)) + ' iterations')
# ax.plot(np.arange(0, 0.5 * len(err_2), 0.5), err_2, label=str(len(U_2)) + ' iterations')
# ax.plot(np.arange(0, 0.25 * len(err_3), 0.25), err_3, label=str(len(U_3)) + ' iterations')
# ax.set_xlabel('Time')
# ax.set_ylabel('Error')
# ax.set_title("Forward Euler")
# ax.legend()
# plt.show()

fig, ax = plt.subplots()  # Create a figure and axes.
ax.plot(np.arange(0, 1 * len(err_1), 1), err_1, label="Preconditioner")
ax.plot(np.arange(0, 0.25 * len(err_3), 0.25), err_3, label="No Preconditioner")
ax.set_xlabel('Time')
ax.set_ylabel('Error')
ax.set_title("Forward Euler")
ax.legend()
plt.show()
