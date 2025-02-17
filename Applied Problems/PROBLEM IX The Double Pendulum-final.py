import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation

g = 9.8
l1 = 1.0
l2 = 1.0
m1 = 1.0
m2 = 1.0

def double_pendulum(t, y):
    theta1, theta1_dot, theta2, theta2_dot = y
    Theta = theta1 - theta2
    F = (m1 + m2) - m2 * np.cos(Theta)**2
    if abs(F) < 1e-6:
        F = 1e-6
    theta1_ddot = (1 / (l1 * F)) * (g * m2 * np.sin(theta2) * np.cos(Theta) - g * (m1 + m2) * np.sin(theta1)
                   - (l2 * m2 * theta2_dot**2 + l1 * m2 * np.cos(Theta) * theta1_dot**2) * np.sin(Theta))
    theta2_ddot = (1 / (l2 * F)) * (g * (m1 + m2) * np.sin(theta1) * np.cos(Theta) - g * (m1 + m2) * np.sin(theta2)
                   + (l1 * (m1 + m2) * theta1_dot**2 + l2 * m2 * np.cos(Theta) * theta2_dot**2) * np.sin(Theta))
    return [theta1_dot, theta1_ddot, theta2_dot, theta2_ddot]

def simulate(ic, t_span, num_points=50000, max_step=None):
    t_eval = np.linspace(t_span[0], t_span[1], num_points)
    kwargs = {'rtol': 1e-10, 'atol': 1e-12}
    if max_step is not None:
        kwargs['max_step'] = max_step
    sol = solve_ivp(double_pendulum, t_span, ic, t_eval=t_eval, method='RK45', **kwargs)
    return sol

# Task 1: Plot θ₁(t) and θ₂(t) for t=0 to 40 s
ic_base = [2.0, 0.0, 1.0, 0.1]
t_span = (0, 40)
sol_base = simulate(ic_base, t_span)
t = sol_base.t
theta1 = sol_base.y[0]
theta2 = sol_base.y[2]
plt.figure()
plt.plot(t, theta1, label=r'$\theta_1$')
plt.plot(t, theta2, label=r'$\theta_2$')
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.legend()
plt.title('Angles vs Time')
plt.show()

# Task 2: Animate the double pendulum motion
x1 = l1 * np.sin(theta1)
y1 = -l1 * np.cos(theta1)
x2 = x1 + l2 * np.sin(theta2)
y2 = y1 - l2 * np.cos(theta2)
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-2.2, 2.2)
ax.set_ylim(-2.2, 2.2)
ax.set_aspect('equal')
line, = ax.plot([], [], 'o-', lw=2)
def init():
    line.set_data([], [])
    return line,
def update(frame):
    i = frame
    line.set_data([0, x1[i], x2[i]], [0, y1[i], y2[i]])
    return line,
ani = FuncAnimation(fig, update, frames=range(0, len(t), 50),
                    init_func=init, blit=True, interval=20)
plt.title('Double Pendulum Animation')
plt.show()

# Task 3: Compare trajectories for slightly different initial conditions
epsilons = [0.0, 1e-3, 1e-5]
sols = []
for eps in epsilons:
    ic = [2.0 + eps, 0.0, 1.0, 0.1]
    sols.append(simulate(ic, t_span))
plt.figure()
for sol, eps in zip(sols, epsilons):
    x1_ = l1 * np.sin(sol.y[0])
    y1_ = -l1 * np.cos(sol.y[0])
    x2_ = x1_ + l2 * np.sin(sol.y[2])
    y2_ = y1_ - l2 * np.cos(sol.y[2])
    plt.plot(x2_, y2_, label=f'ε = {eps}')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Trace of m₂ with Slightly Different ICs')
plt.show()

# Task 4: Estimate the Lyapunov exponent using ε = 1e-5
sol1 = sols[0]
sol2 = sols[2]
state_diff = sol2.y - sol1.y
delta = np.linalg.norm(state_diff, axis=0)
delta0 = delta[0]
log_delta = np.log(delta / delta0)
mask = sol1.t <= 20  # using early-time exponential growth
coef = np.polyfit(sol1.t[mask], log_delta[mask], 1)
lyapunov = coef[0]
plt.figure()
plt.plot(sol1.t, log_delta, label=r'$\ln(\delta/\delta_0)$')
plt.plot(sol1.t[mask], np.polyval(coef, sol1.t[mask]), '--',
         label=f'Fit: λ = {lyapunov:.4f}')
plt.xlabel('Time (s)')
plt.ylabel(r'$\ln(\delta/\delta_0)$')
plt.legend()
plt.title('Lyapunov Exponent Estimation')
plt.show()
print(f"Estimated Lyapunov exponent: {lyapunov:.4f} 1/s")

# Task 5: Effect of mass ratio m₁/m₂ on chaos
mass_ratios = [1, 2, 5, 10, 20]
lyapunovs = []
orig_m1, orig_m2 = m1, m2
for ratio in mass_ratios:
    m1 = ratio
    m2 = 1.0
    sol_base = simulate([2.0, 0.0, 1.0, 0.1], t_span)
    sol_pert = simulate([2.0 + 1e-5, 0.0, 1.0, 0.1], t_span)
    diff = np.linalg.norm(sol_pert.y - sol_base.y, axis=0)
    d0 = diff[0]
    log_diff = np.log(diff / d0)
    mask = sol_base.t <= 20
    coef = np.polyfit(sol_base.t[mask], log_diff[mask], 1)
    lyapunovs.append(coef[0])
m1, m2 = orig_m1, orig_m2
plt.figure()
plt.plot(mass_ratios, lyapunovs, 'o-')
plt.xlabel('m₁/m₂')
plt.ylabel('Lyapunov exponent (1/s)')
plt.title('Mass Ratio vs. Lyapunov Exponent')
plt.show()

# Task 6: Sensitivity to integration stepsize (max_step)
stepsizes = [0.001, 0.01, 0.1]
trajectories = []
for h in stepsizes:
    trajectories.append(simulate([2.0, 0.0, 1.0, 0.1], t_span, max_step=h))
plt.figure()
for sol, h in zip(trajectories, stepsizes):
    plt.plot(sol.y[0], sol.y[2], label=f'h = {h}')
plt.xlabel(r'$\theta_1$')
plt.ylabel(r'$\theta_2$')
plt.legend()
plt.title('Phase Plot for Different Stepsizes')
plt.show()
