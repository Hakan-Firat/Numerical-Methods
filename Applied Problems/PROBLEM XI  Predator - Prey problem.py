import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def p_p(t, z, alpha, beta, delta, gamma):
    x, y = z
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return [dxdt, dydt]

params = (0.1, 0.02, 0.01, 0.1)
initial_conditions = [40, 9]
t_span = (0, 200)
t_eval = np.linspace(*t_span, 1000)

sol = solve_ivp(p_p, t_span, initial_conditions, args=params, t_eval=t_eval)

plt.figure()
plt.plot(sol.t, sol.y[0], label="Prey")
plt.plot(sol.t, sol.y[1], label="Predator")
plt.xlabel("Time")
plt.ylabel("Population")
plt.legend()
plt.show()

plt.figure()
plt.plot(sol.y[0], sol.y[1])
plt.xlabel("Prey Population")
plt.ylabel("Predator Population")
plt.title("Phase Diagram")
plt.show()
