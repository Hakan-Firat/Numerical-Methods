

# ----------------------------------------------------------------

# used solve_ivp which is an explicit RK method with adaptive step size

# ----------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp

g = 9.8
l = 1.0
t_span = (0, 15)
y0 = [np.pi/4, 0]

def pendulum(t, y):
    return [y[1], -g/l * np.sin(y[0])]

sol = solve_ivp(pendulum, t_span, y0, t_eval=np.linspace(0, 15, 300))

t = sol.t
num_theta = sol.y[0]

# analytical solution with small angle approximation
analytical_theta = (np.pi / 4) * np.cos(np.sqrt(g/l) * t)

plt.plot(t, num_theta, label='Numerical')
plt.plot(t, analytical_theta, '--', label='Analytical')
plt.xlabel('t (s)')
plt.ylabel(r'$\theta(t)$')
plt.legend()
plt.show()

fig, ax = plt.subplots()
ax.set_xlim(-l, l)
ax.set_ylim(-1, l)
line, = ax.plot([], [], 'o-', lw=2)

def init():
    line.set_data([], [])
    return line,

def update(frame):
    x = l * np.sin(num_theta[frame])
    y = -l * np.cos(num_theta[frame])
    line.set_data([0, x], [0, y])
    return line,

ani = animation.FuncAnimation(fig, update, frames=len(t), init_func=init, blit=True, interval=50)
plt.show()