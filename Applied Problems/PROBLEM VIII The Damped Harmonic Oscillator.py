import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp

m, k, b = 1.0, 1.0, 0.2
t_span = (0, 15)
y0 = [1.0, 0.0]

def damped_oscillator(t, y):
    return [y[1], -k/m * y[0] - b/m * y[1]]

sol = solve_ivp(damped_oscillator, t_span, y0, t_eval=np.linspace(0, 15, 300))

t = sol.t
x = sol.y[0]
v = sol.y[1]

plt.plot(t, x)
plt.xlabel('t (s)')
plt.ylabel('x(t)')
plt.title('Damped Harmonic Oscillator')
plt.show()

plt.plot(x, v)
plt.xlabel('x')
plt.ylabel('v')
plt.title('Phase Diagram')
plt.show()

# ----------------------------------------------------------------
#                   Comparison
# ----------------------------------------------------------------

g = 9.8
l = 1.0
t_span = (0, 15)
y0 = [np.pi/4, 0]

def pendulum(t, y):
    return [y[1], -g/l * np.sin(y[0])]

sol = solve_ivp(pendulum, t_span, y0, t_eval=np.linspace(0, 15, 300))

t = sol.t
num_theta = sol.y[0]
analytical_theta = (np.pi / 4) * np.cos(np.sqrt(g/l) * t)

plt.figure(figsize=(10, 4))
plt.plot(t, num_theta, label='Numerical (Pendulum)')
plt.plot(t, analytical_theta, '--', label='Analytical (Pendulum)')
plt.xlabel('t (s)')
plt.ylabel(r'$\theta(t)$')
plt.legend()
plt.title('Pendulum Motion Comparison')
plt.show()

fig, ax = plt.subplots()
ax.set_xlim(-l, l)
ax.set_ylim(-l, l)
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

# Damped Harmonic Oscillator
m, k, b = 1.0, 1.0, 0.2
t_span = (0, 15)
y0 = [1.0, 0.0]

def damped_oscillator(t, y):
    return [y[1], -k/m * y[0] - b/m * y[1]]

sol = solve_ivp(damped_oscillator, t_span, y0, t_eval=np.linspace(0, 15, 300))

t = sol.t
x = sol.y[0]
v = sol.y[1]

plt.figure(figsize=(10, 4))
plt.plot(t, x, label='Damped Oscillator')
plt.xlabel('t (s)')
plt.ylabel('x(t)')
plt.title('Damped Harmonic Oscillator')
plt.legend()
plt.show()

plt.figure(figsize=(6, 6))
plt.plot(x, v, label='Damped Oscillator')
plt.plot(num_theta, np.gradient(num_theta, t), label='Pendulum')
plt.xlabel('x or θ')
plt.ylabel('v or dθ/dt')
plt.title('Phase Diagram Comparison')
plt.legend()
plt.show()
