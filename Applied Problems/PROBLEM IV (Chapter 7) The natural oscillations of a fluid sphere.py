import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import brentq
import matplotlib.pyplot as plt

n = 1.5
G = 1.0
K = 1.0
rho_c = 1.0
Gamma1 = 1 + 1/n
a = np.sqrt((n+1)*K/(4*np.pi*G))

# Lane–Emden ODE; clamp theta to nonnegative values when computing theta**n
def lane_emden(xi, y):
    theta, phi = y
    theta_pos = max(theta, 0)
    if xi == 0:
        return [phi, 0]
    else:
        return [phi, -2/xi * phi - theta_pos**n]

# Event: stop when theta=0 (from above)
def event_zero(xi, y):
    return y[0]
event_zero.terminal = True
event_zero.direction = -1

xi_min = 1e-6
y0 = [1 - xi_min**2/6, -xi_min/3]
xi_span = (xi_min, 10)
sol = solve_ivp(
    lane_emden, xi_span, y0, events=event_zero,
    max_step=1e-4, rtol=1e-10, atol=1e-12, dense_output=True
)
if sol.t_events[0].size == 0:
    raise ValueError("Zero crossing not found")
xi_R = sol.t_events[0][0]
R = a * xi_R

# interpolators for theta and phi as functions of xi
theta_interp = interp1d(sol.t, sol.y[0], kind='cubic', fill_value="extrapolate")
phi_interp = interp1d(sol.t, sol.y[1], kind='cubic', fill_value="extrapolate")

#  clamp negative theta evaluations to zero
def theta_at_r(r):
    th = theta_interp(r/a)
    return th if th > 0 else 0

def phi_at_r(r):
    return phi_interp(r/a)

# Pressure and Density 
def rho(r):
    th = theta_at_r(r)
    return rho_c * (th**n)

def P(r):
    th = theta_at_r(r)
    return K * rho_c**(1+1/n) * (th**(n+1))

# Pressure and Density Gradients
def dP_dr(r):
    th = theta_at_r(r)
    ph = phi_at_r(r)
    return K * rho_c**(1+1/n) * (n+1)/a * (th**n) * ph

def drho_dr(r):
    th = theta_at_r(r)
    ph = phi_at_r(r)
    return rho_c * n/a * (th**(n-1)) * ph if th > 0 else 0

# Radial Oscillation Equation
def osc_ode(r, Y, omega):
    xi_val, dxi = Y
    p = P(r)
    dpdr = dP_dr(r)
    dens = rho(r)
    # Avoid division by zero at the surface
    if p == 0:
        term1 = term2 = 0
    else:
        term1 = 2/r + dpdr/(Gamma1*p)
        term2 = (omega**2 * dens)/(Gamma1*p) - 4/r**2
    return [dxi, -term1*dxi - term2*xi_val]

# Initial Conditions for Radial Oscillations
s = (np.sqrt(17)-1)/2
r_osc_min = 1e-6
Y0 = [r_osc_min**s, s*r_osc_min**(s-1)]

# Integrate the oscillation equation with given w
def shoot(omega):
    sol_osc = solve_ivp(
        lambda r, Y: osc_ode(r, Y, omega),
        [r_osc_min, R], Y0, max_step=1e-4, rtol=1e-10, atol=1e-12
    )
    #Extracting the Boundary Condition at 𝑟=𝑅
    xi_R_val = sol_osc.y[0, -1]
    dxi_R_val = sol_osc.y[1, -1]
    rhoR = rho(R)
    drhoR = drho_dr(R)
    # Returning an omega value that is zero when when w is correct
    return dxi_R_val + (drhoR/rhoR)*xi_R_val if rhoR != 0 else dxi_R_val

# Find sign change in shooting function for eigenfrequency range, one is positive and one is negative, meaning there is a eigenfrequency in between.
omega1, omega2 = 0.1, 10.0
f1, f2 = shoot(omega1), shoot(omega2)
if np.isnan(f1) or np.isnan(f2):
    raise ValueError("Shooting function returned NaN")
if f1 * f2 > 0:
    raise ValueError("No sign change found in shooting function")

omega_eigen = brentq(shoot, omega1, omega2)
print(f"Eigenfrequency: omega = {omega_eigen:.5f}")

# Final radial eigenfunction solution
sol_final = solve_ivp(
    lambda r, Y: osc_ode(r, Y, omega_eigen),
    [r_osc_min, R], Y0, max_step=1e-4, rtol=1e-10, atol=1e-12, dense_output=True
)

# Plot radial eigenfunction	solution 1000 points
r_plot = np.linspace(r_osc_min, R, 1000)
xi_plot = sol_final.sol(r_plot)[0]

plt.plot(r_plot, xi_plot)
plt.xlabel("r")
plt.ylabel(r"$\xi(r)$")
plt.title(f"Radial eigenfunction, omega = {omega_eigen:.5f}")
plt.show()
