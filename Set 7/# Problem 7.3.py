import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# matrix method
def solve_by_matrix_method(x1, x2, u1, u2, n=100):
    x_vals = np.linspace(x1, x2, n)
    dx = x_vals[1] - x_vals[0]
    
    A = np.zeros((n, n))
    b = np.zeros(n)
    
    for i in range(1, n - 1):
        A[i, i - 1] = 1 / dx**2
        A[i, i] = -2 / dx**2 + 2
        A[i, i + 1] = 1 / dx**2
    
    A[0, 0] = 1
    b[0] = u1
    A[-1, -1] = 1
    b[-1] = u2
    
    y_vals = np.linalg.solve(A, b)
    
    return x_vals, y_vals

# shooting method with iteration tracking
def shooting_method_ode(x, y):
    return [y[1], 2 * y[0]]

def shooting_root(v, x1, x2, u1, u2, iterates):
    sol = solve_ivp(shooting_method_ode, [x1, x2], [u1, v], t_eval=np.linspace(x1, x2, 100), method='RK45')
    iterates.append((sol.t, sol.y[0]))
    return sol.y[0][-1] - u2

def solve_by_shooting(x1, x2, u1, u2, v_guess):
    iterates = []
    sol = root_scalar(shooting_root, args=(x1, x2, u1, u2, iterates), bracket=[v_guess - 1, v_guess + 1], method='brentq')
    x_vals = np.linspace(x1, x2, 100)
    sol_ivp = solve_ivp(shooting_method_ode, [x1, x2], [u1, sol.root], t_eval=x_vals, method='RK45')
    iterates.append((sol_ivp.t, sol_ivp.y[0]))
    return x_vals, sol_ivp.y[0], iterates

x1, x2 = 0, 1
u1, u2 = 1.2, 0.9
v_guess = -1

x_vals_matrix, u_vals_matrix = solve_by_matrix_method(x1, x2, u1, u2)
x_vals_shooting, u_vals_shooting, iterates = solve_by_shooting(x1, x2, u1, u2, v_guess)

print("Solution using Matrix Method:")
for x, u in zip(x_vals_matrix, u_vals_matrix):
    print(f"x = {x:.4f}, u = {u:.4f}")

print("\nSolution using Shooting Method:")
for x, u in zip(x_vals_shooting, u_vals_shooting):
    print(f"x = {x:.4f}, u = {u:.4f}")

# Compare
plt.figure(figsize=(8, 6))
plt.plot(x_vals_matrix, u_vals_matrix, label='Matrix Method Solution', linestyle='--', marker='o')
plt.plot(x_vals_shooting, u_vals_shooting, label='Shooting Method Solution', linestyle='-', marker='x')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.legend()
plt.title('Comparison of Matrix and Shooting Method Solutions for BVP')
plt.show()

# Create animation
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlabel('x')
ax.set_ylabel('u(x)')
ax.set_title('Shooting Method Iterations')
line, = ax.plot([], [], lw=2)
text = ax.text(0.05, 0.9, '', transform=ax.transAxes, fontsize=12, color='red')
ax.set_xlim(x1, x2)
ax.set_ylim(min(min(u) for _, u in iterates) - 0.5, max(max(u) for _, u in iterates) + 0.5)

def init():
    line.set_data([], [])
    text.set_text('')
    return line, text

def update(frame):
    x_it, u_it = iterates[frame]
    line.set_data(x_it, u_it)
    text.set_text(f'Iteration {frame + 1}')
    return line, text

ani = animation.FuncAnimation(fig, update, frames=len(iterates), init_func=init, blit=True, repeat=True, interval=500)
plt.show()





# ---------------------------------------------------------------- Compared to analytical solution

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def solve_by_matrix_method(x1, x2, u1, u2, n=100):
    x_vals = np.linspace(x1, x2, n)
    dx = x_vals[1] - x_vals[0]
    
    A = np.zeros((n, n))
    b = np.zeros(n)
    
    # Fill the matrix using finite differences
    for i in range(1, n - 1):
        A[i, i - 1] = 1 / dx**2
        A[i, i] = -2 / dx**2 + 2
        A[i, i + 1] = 1 / dx**2
    
    # Boundary conditions
    A[0, 0] = 1
    b[0] = u1
    A[-1, -1] = 1
    b[-1] = u2
    
    y_vals = np.linalg.solve(A, b)
    
    return x_vals, y_vals

# shooting method with iteration tracking
def shooting_method_ode(x, y):
    return [y[1], 2 * y[0]]

def shooting_root(v, x1, x2, u1, u2, iterates):
    sol = solve_ivp(shooting_method_ode, [x1, x2], [u1, v], t_eval=np.linspace(x1, x2, 100), method='RK45')
    iterates.append((sol.t, sol.y[0]))
    return sol.y[0][-1] - u2

def solve_by_shooting(x1, x2, u1, u2, v_guess):
    iterates = []
    sol = root_scalar(shooting_root, args=(x1, x2, u1, u2, iterates), bracket=[v_guess - 1, v_guess + 1], method='brentq')
    x_vals = np.linspace(x1, x2, 100)
    sol_ivp = solve_ivp(shooting_method_ode, [x1, x2], [u1, sol.root], t_eval=x_vals, method='RK45')
    iterates.append((sol_ivp.t, sol_ivp.y[0]))
    return x_vals, sol_ivp.y[0], iterates

# Exact analytical solution
def exact_solution(x, u1, u2, x1, x2):
    sqrt2 = np.sqrt(2)
    C1 = (u2 - u1 * np.exp(-sqrt2 * x2)) / (np.exp(sqrt2 * x2) - np.exp(-sqrt2 * x2))
    C2 = u1 - C1
    return C1 * np.exp(sqrt2 * x) + C2 * np.exp(-sqrt2 * x)

x1, x2 = 0, 1
u1, u2 = 1.2, 0.9
v_guess = -1

x_vals_matrix, u_vals_matrix = solve_by_matrix_method(x1, x2, u1, u2)
x_vals_shooting, u_vals_shooting, iterates = solve_by_shooting(x1, x2, u1, u2, v_guess)
x_vals_exact = np.linspace(x1, x2, 100)
u_vals_exact = exact_solution(x_vals_exact, u1, u2, x1, x2)

# Print results
print("Solution using Matrix Method:")
for x, u in zip(x_vals_matrix, u_vals_matrix):
    print(f"x = {x:.4f}, u = {u:.4f}")

print("\nSolution using Shooting Method:")
for x, u in zip(x_vals_shooting, u_vals_shooting):
    print(f"x = {x:.4f}, u = {u:.4f}")

print("\nExact Analytical Solution:")
for x, u in zip(x_vals_exact, u_vals_exact):
    print(f"x = {x:.4f}, u = {u:.4f}")

# Compare all methods
plt.figure(figsize=(8, 6))
plt.plot(x_vals_matrix, u_vals_matrix, label='Matrix Method Solution', linestyle='--', marker='o')
plt.plot(x_vals_shooting, u_vals_shooting, label='Shooting Method Solution', linestyle='-', marker='x')
plt.plot(x_vals_exact, u_vals_exact, label='Exact Solution', linestyle='-', linewidth=2)
plt.xlabel('x')
plt.ylabel('u(x)')
plt.legend()
plt.title('Comparison of Matrix, Shooting, and Exact Solutions for BVP')
plt.show()

# Create animation
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlabel('x')
ax.set_ylabel('u(x)')
ax.set_title('Shooting Method Iterations')
line, = ax.plot([], [], lw=2)
text = ax.text(0.05, 0.9, '', transform=ax.transAxes, fontsize=12, color='red')
ax.set_xlim(x1, x2)
ax.set_ylim(min(min(u) for _, u in iterates) - 0.5, max(max(u) for _, u in iterates) + 0.5)

def init():
    line.set_data([], [])
    text.set_text('')
    return line, text

def update(frame):
    x_it, u_it = iterates[frame]
    line.set_data(x_it, u_it)
    text.set_text(f'Iteration {frame + 1}')
    return line, text

ani = animation.FuncAnimation(fig, update, frames=len(iterates), init_func=init, blit=True, repeat=True, interval=500)
plt.show()



# ---------------------------------------------------------------- exact solution from wolframalpha

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def solve_by_matrix_method(x1, x2, u1, u2, n=100):
    x_vals = np.linspace(x1, x2, n)
    dx = x_vals[1] - x_vals[0]
    
    A = np.zeros((n, n))
    b = np.zeros(n)
    
    for i in range(1, n - 1):
        A[i, i - 1] = 1 / dx**2
        A[i, i] = -2 / dx**2 + 2
        A[i, i + 1] = 1 / dx**2
    
    A[0, 0] = 1
    b[0] = u1
    A[-1, -1] = 1
    b[-1] = u2
    
    y_vals = np.linalg.solve(A, b)
    
    return x_vals, y_vals

def shooting_method_ode(x, y):
    return [y[1], 2 * y[0]]

def shooting_root(v, x1, x2, u1, u2, iterates):
    sol = solve_ivp(shooting_method_ode, [x1, x2], [u1, v], t_eval=np.linspace(x1, x2, 100), method='RK45')
    iterates.append((sol.t, sol.y[0]))
    return sol.y[0][-1] - u2

def solve_by_shooting(x1, x2, u1, u2, v_guess):
    iterates = []
    sol = root_scalar(shooting_root, args=(x1, x2, u1, u2, iterates), bracket=[v_guess - 1, v_guess + 1], method='brentq')
    x_vals = np.linspace(x1, x2, 100)
    sol_ivp = solve_ivp(shooting_method_ode, [x1, x2], [u1, sol.root], t_eval=x_vals, method='RK45')
    iterates.append((sol_ivp.t, sol_ivp.y[0]))
    return x_vals, sol_ivp.y[0], iterates

def exact_solution_wolfram(x):
    return 1.04283 * np.exp(-np.sqrt(2) * x) + 0.157168 * np.exp(np.sqrt(2) * x)

x1, x2 = 0, 1
u1, u2 = 1.2, 0.9
v_guess = -1

x_vals_matrix, u_vals_matrix = solve_by_matrix_method(x1, x2, u1, u2)
x_vals_shooting, u_vals_shooting, iterates = solve_by_shooting(x1, x2, u1, u2, v_guess)
x_vals_exact = np.linspace(x1, x2, 100)
u_vals_exact = exact_solution_wolfram(x_vals_exact)

print("Solution using Matrix Method:")
for x, u in zip(x_vals_matrix, u_vals_matrix):
    print(f"x = {x:.4f}, u = {u:.4f}")

print("\nSolution using Shooting Method:")
for x, u in zip(x_vals_shooting, u_vals_shooting):
    print(f"x = {x:.4f}, u = {u:.4f}")

print("\nExact Analytical Solution (Wolfram Alpha):")
for x, u in zip(x_vals_exact, u_vals_exact):
    print(f"x = {x:.4f}, u = {u:.4f}")

# Compare all methods
plt.figure(figsize=(8, 6))
plt.plot(x_vals_matrix, u_vals_matrix, label='Matrix Method Solution', linestyle='--', marker='o')
plt.plot(x_vals_shooting, u_vals_shooting, label='Shooting Method Solution', linestyle='-', marker='x')
plt.plot(x_vals_exact, u_vals_exact, label='Exact Solution (Wolfram Alpha)', linestyle='-', linewidth=2)
plt.xlabel('x')
plt.ylabel('u(x)')
plt.legend()
plt.title('Comparison of Matrix, Shooting, and Exact Solutions for BVP')
plt.show()

# Create animation
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlabel('x')
ax.set_ylabel('u(x)')
ax.set_title('Shooting Method Iterations')
line, = ax.plot([], [], lw=2)
text = ax.text(0.05, 0.9, '', transform=ax.transAxes, fontsize=12, color='red')
ax.set_xlim(x1, x2)
ax.set_ylim(min(min(u) for _, u in iterates) - 0.5, max(max(u) for _, u in iterates) + 0.5)

def init():
    line.set_data([], [])
    text.set_text('')
    return line, text

def update(frame):
    x_it, u_it = iterates[frame]
    line.set_data(x_it, u_it)
    text.set_text(f'Iteration {frame + 1}')
    return line, text

ani = animation.FuncAnimation(fig, update, frames=len(iterates), init_func=init, blit=True, repeat=True, interval=500)
plt.show()


# The shooting method allings well with the exact solution of the differential problem, and from my guess, this is due to shooting method finds the slope by itirating compared to matrix methods which discretizes the domain and approximates the second derivative using a finite difference scheme. 