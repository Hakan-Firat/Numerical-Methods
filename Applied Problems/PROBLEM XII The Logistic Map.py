import numpy as np
import matplotlib.pyplot as plt

def logistic(x, r):
    return r * x * (1 - x)

def iterate(x0, r, n):
    xs = np.empty(n)
    xs[0] = x0
    for i in range(1, n):
        xs[i] = logistic(xs[i-1], r)
    return xs

def lyapunov(r, x0, n):
    x = x0
    L = 0.0
    for _ in range(n):
        x = logistic(x, r)
        L += np.log(abs(r * (1 - 2*x)))
    return L / n

def fixed_points(r):
    pts = [0]
    if r > 1:
        pts.append((r - 1) / r)
    return pts

def period2_points(r):
    if r > 3:
        disc = np.sqrt((r - 3) * (r + 1))
        return [(r + 1 - disc) / (2 * r), (r + 1 + disc) / (2 * r)]
    return []

# Simulation for r=3.3
r_val = 3.3
x0 = 0.2
n_iter = 100
xs = iterate(x0, r_val, n_iter)
print("Fixed points:", fixed_points(r_val))
print("Period-2 points:", period2_points(r_val))
print("Lyapunov exponent:", lyapunov(r_val, x0, 1000))

plt.figure()
plt.plot(range(n_iter), xs, 'o-')
plt.xlabel("Iteration")
plt.ylabel("x")
plt.title(f"Logistic map (r={r_val})")
plt.show()

# Bifurcation diagram
r_vals = np.linspace(2.5, 4.0, 1000)
iterations = 500
last = 100
x = 0.5 * np.ones_like(r_vals)
rs, xs_plot = [], []
for i in range(iterations):
    x = logistic(x, r_vals)
    if i >= (iterations - last):
        rs.extend(r_vals)
        xs_plot.extend(x)
plt.figure(figsize=(8,6))
plt.plot(rs, xs_plot, ',k', alpha=0.25)
plt.xlabel("r")
plt.ylabel("x")
plt.title("Bifurcation Diagram")
plt.show()

# Lyapunov exponent vs r
r_vals = np.linspace(2.5, 4.0, 1000)
L = np.zeros_like(r_vals)
x = 0.5 * np.ones_like(r_vals)
n_lyap = 1000
for _ in range(n_lyap):
    x = logistic(x, r_vals)
    L += np.log(abs(r_vals * (1 - 2*x)))
L /= n_lyap
plt.figure()
plt.plot(r_vals, L, 'r')
plt.axhline(0, color='k', lw=0.5)
plt.xlabel("r")
plt.ylabel("Lyapunov Exponent")
plt.title("Lyapunov Exponent vs r")
plt.show()


def logistic(x, r):
    return r * x * (1 - x)

x = np.linspace(0, 1, 500)
r_values = [1, 2, 3, 4]

plt.figure(figsize=(8,6))
for r in r_values:
    plt.plot(x, logistic(x, r), label=f"r={r}")
plt.plot(x, x, 'k--', lw=1)
plt.xlabel("x_n")
plt.ylabel("x_(n+1)")
plt.title("Logistic Map: f(x)=r*x*(1-x)")
plt.legend()
plt.show()
