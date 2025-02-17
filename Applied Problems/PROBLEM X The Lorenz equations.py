import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp
from scipy.stats import linregress

# Lorenz system
def lorenz(t, state, sigma, r, b):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (r - z) - y
    dz = x * y - b * z
    return [dx, dy, dz]

# Parameters
sigma, r, b = 10, 28, 8/3
init_cond = [1, 1, 1]
perturbed_cond = [1 + 1e-9, 1, 1]

# Time span
t_span = (0, 50)
t_eval = np.linspace(t_span[0], t_span[1], 10000)

# Solve
sol1 = solve_ivp(lorenz, t_span, init_cond, args=(sigma, r, b), t_eval=t_eval)
sol2 = solve_ivp(lorenz, t_span, perturbed_cond, args=(sigma, r, b), t_eval=t_eval)

# Calculate Lyapunov exponent
start_index = np.searchsorted(sol1.t, 10)  
end_index = np.searchsorted(sol1.t, 40)   
delta_x = np.abs(sol1.y[0] - sol2.y[0])
log_delta_x = np.log(delta_x[start_index:end_index])
time_fit = sol1.t[start_index:end_index]

# slope of the logarithmic divergence
slope, _, _, _, _ = linregress(time_fit, log_delta_x)
lyapunov_exponent = slope

# Animation
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-20, 20])
ax.set_ylim([-30, 30])
ax.set_zlim([0, 50])
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
line1, = ax.plot([], [], [], lw=0.5, label="Trajectory A")
line2, = ax.plot([], [], [], lw=0.5, label="Trajectory B", color='r')
ax.legend()

def update(num):
    step = 100  # animation speed
    line1.set_data(sol1.y[0][:num * step], sol1.y[1][:num * step])
    line1.set_3d_properties(sol1.y[2][:num * step])
    line2.set_data(sol2.y[0][:num * step], sol2.y[1][:num * step])
    line2.set_3d_properties(sol2.y[2][:num * step])
    return line1, line2

ani = animation.FuncAnimation(fig, update, frames=len(sol1.t)//10, interval=1, blit=False)
plt.show()


# sensitivity, the difference between the two initial conditions
plt.figure(figsize=(8, 5))
plt.plot(sol1.t, sol1.y[0] - sol2.y[0], label='Δx(t)')
plt.xlabel("Time")
plt.ylabel("Difference in x(t)")
plt.title("Chaotic Sensitivity to Initial Conditions")
plt.legend()
plt.show()


plt.figure(figsize=(8, 5))
plt.plot(time_fit, log_delta_x, label=f'λ ≈ {lyapunov_exponent:.3f}')
plt.xlabel("Time")
plt.ylabel("log(Δx(t))")
plt.title("Lyapunov Exponent Approximation")
plt.legend()
plt.show()
