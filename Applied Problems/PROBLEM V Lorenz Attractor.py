import numpy as np
import matplotlib.pyplot as plt
'''
def rk4_step(f, y, t, h, params):
    k1 = f(y, t, params)
    k2 = f(y + 0.5 * h * k1, t + 0.5 * h, params)
    k3 = f(y + 0.5 * h * k2, t + 0.5 * h, params)
    k4 = f(y + h * k3, t + h, params)
    return y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

def lorenz(y, t, params):
    sigma, R, beta = params
    x, y, z = y
    return np.array([sigma * (y - x), -x * z + R * x - y, x * y - beta * z])

def integrate(f, y0, t_span, h, params):
    t_values = np.arange(t_span[0], t_span[1], h)
    y_values = np.zeros((len(t_values), len(y0)))
    y_values[0] = y0
    for i in range(1, len(t_values)):
        y_values[i] = rk4_step(f, y_values[i-1], t_values[i-1], h, params)
        print(y_values[i])
    return t_values, y_values

# Parameters and initial conditions
params = (10, 28, 8/3)
y0 = np.array([5.0, 5.0, 5.0])
t_span = (0, 50)
h = 1e-3

# Solve
t_values, y_values = integrate(lorenz, y0, t_span, h, params)

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(y_values[:, 0], y_values[:, 1], y_values[:, 2], linewidth=0.5)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
'''



# ----------------------------------------------------------------


#Double Trajectories and Chaos - part a

import numpy as np
import matplotlib.pyplot as plt

def rk4_step(f, y, t, h, params):
    k1 = f(y, t, params)
    k2 = f(y + 0.5 * h * k1, t + 0.5 * h, params)
    k3 = f(y + 0.5 * h * k2, t + 0.5 * h, params)
    k4 = f(y + h * k3, t + h, params)
    return y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

def lorenz(y, t, params):
    sigma, R, beta = params
    x, y, z = y
    return np.array([sigma * (y - x), -x * z + R * x - y, x * y - beta * z])

def integrate(f, y0, t_span, h, params):
    t_values = np.arange(t_span[0], t_span[1], h)
    y_values = np.zeros((len(t_values), len(y0)))
    y_values[0] = y0
    for i in range(1, len(t_values)):
        y_values[i] = rk4_step(f, y_values[i-1], t_values[i-1], h, params)
    return t_values, y_values

def compute_distance(y1, y2):
    return np.linalg.norm(y1 - y2, axis=1)

# Parameters and initial conditions
params = (10, 28, 8/3)
y0_1 = np.array([5.0, 5.0, 5.0])
y0_2 = np.array([5.0, 5.0, 5.0 + 1e-5])
t_span = (0, 20)
h = 1e-3

# Solve for both initial conditions
t_values, y_values_1 = integrate(lorenz, y0_1, t_span, h, params)
_, y_values_2 = integrate(lorenz, y0_2, t_span, h, params)

distances = compute_distance(y_values_1, y_values_2)

# Plot log(d) vs. x
plt.figure()
plt.plot(y_values_1[:, 0], np.log(distances))
plt.xlabel('X')
plt.ylabel('log(d)')
plt.show()
'''
# part b

# Compare r(20) for different step sizes
h_small = 1e-6
h_large = 5e-4

_, y_values_small = integrate(lorenz, y0_1, (0, 20), h_small, params)
_, y_values_large = integrate(lorenz, y0_1, (0, 20), h_large, params)

r_20_small = y_values_small[-1]
r_20_large = y_values_large[-1]

print("Final position with h = 10^-6:", r_20_small)
print("Final position with h = 5 × 10^-4:", r_20_large)
print("Difference:", np.linalg.norm(r_20_small - r_20_large))
'''
# part c

y0_3 = np.array([5.0, 5.0, 5.0 + 5e-15])
t_span_long = (0, 50)

t_values_long, y_values_3 = integrate(lorenz, y0_3, t_span_long, h, params)
_, y_values_4 = integrate(lorenz, y0_1, t_span_long, h, params)

distances_long = compute_distance(y_values_3, y_values_4)

plt.figure()
plt.plot(y_values_3[:, 0], np.log(distances_long))
plt.xlabel('X')
plt.ylabel('log(d) for t=50')
plt.show()


# exponential sensitivity with λ which is the Lyapunov exponent. d(t)=d_0 * exp(λ*t) which gives us d at 50 about 10^5 difference. with largest largest Lyapunov exponent 0.91 for classical Lorenz system

# difference
final_distance_50 = distances_long[-1]
print("Final distance at t=50:", final_distance_50)
print("Log of final distance at t=50:", np.log(final_distance_50))

# Compare with theoretical estimate
theoretical_d50 = (5e-15) * np.exp(0.91 * 50)
print("Theoretical estimate of d(50):", theoretical_d50)
print("Order of magnitude difference:", theoretical_d50 / final_distance_50)
