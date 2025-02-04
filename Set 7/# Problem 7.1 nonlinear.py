# Problem 7.1 nonlinear

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def shooting_method_ode(x, y, g):
    u, v = y  # y[0] = u, y[1] = u'
    return [v, g(u, v, x)]

def solve_ivp_rk4(g, x1, x2, u1, v, step_size=1e-4):
    x_vals = np.arange(x1, x2 + step_size, step_size)
    sol = solve_ivp(shooting_method_ode, [x1, x2], [u1, v], args=(g,), t_eval=x_vals, method='RK45')
    return sol.t, sol.y[0]

# adjust v
def shooting_root(v, g, x1, x2, u1, u2, iterates):
    x_vals, u_vals = solve_ivp_rk4(g, x1, x2, u1, v)
    iterates.append((x_vals, u_vals))
    result = u_vals[-1] - u2  # Difference between computed and target boundary condition
    print(f"v: {v}, shooting_root: {result}")  # Debugging output
    return result

# shooting method
def solve_by_shooting(g, x1, x2, u1, u2, v_guess):
    iterates = []
    
    v_lower, v_upper = v_guess - 1, v_guess + 1
    f_lower = shooting_root(v_lower, g, x1, x2, u1, u2, iterates)
    f_upper = shooting_root(v_upper, g, x1, x2, u1, u2, iterates)
    
    while f_lower * f_upper > 0: 
        v_lower -= 1
        v_upper += 1
        f_lower = shooting_root(v_lower, g, x1, x2, u1, u2, iterates)
        f_upper = shooting_root(v_upper, g, x1, x2, u1, u2, iterates)
    
    sol = root_scalar(shooting_root, args=(g, x1, x2, u1, u2, iterates), bracket=[v_lower, v_upper], method='brentq', xtol=1e-4)
    
    if sol.converged:
        v_corrected = sol.root
        x_vals, u_vals = solve_ivp_rk4(g, x1, x2, u1, v_corrected)
        iterates.append((x_vals, u_vals))
        print(f"Final estimated initial derivative: {v_corrected}")
        return x_vals, u_vals, iterates, v_corrected
    else:
        raise ValueError("Root finding did not converge")

def nonlinear_g(u, v, x):
    return (1 - x / 5) * u * v + x

x1, x2 = 1, 3
u1, u2 = 2, -1
v_guess = -2 

x_vals, u_vals, iterates, v_final = solve_by_shooting(nonlinear_g, x1, x2, u1, u2, v_guess)
target_nonlinear_v = -2.0161
print(f"Computed initial derivative v'(1) for nonlinear BVP: {v_final} (Target: {target_nonlinear_v})")

# Create animation
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlabel('x')
ax.set_ylabel('u(x)')
ax.set_title('Shooting Method Iterations (Nonlinear)')
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


# Plot the solutions for each iteration
plt.figure(figsize=(8, 6))
for i, (x_it, u_it) in enumerate(iterates):
    plt.plot(x_it, u_it, label=f'Iteration {i+1}', alpha=0.7)
plt.xlabel('x')
plt.ylabel('u(x)')
plt.legend()
plt.title('Shooting Method Iterations, nonlinear')
plt.show()
