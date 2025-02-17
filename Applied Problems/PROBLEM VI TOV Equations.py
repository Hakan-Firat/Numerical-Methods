import numpy as np
import matplotlib.pyplot as plt

# RK4 
def rk4_step(f, r, y, h, args):
    k1 = f(r, y, *args)
    k2 = f(r + 0.5*h, y + 0.5*h*k1, *args)
    k3 = f(r + 0.5*h, y + 0.5*h*k2, *args)
    k4 = f(r + h, y + h*k3, *args)
    return y + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

# derivatives: dP/dr and dm/dr, P = κ ρ₀^Γ, ρ = ρ₀ + P/(Γ-1) with ρ₀ = (P/κ)^(1/Γ)
def tov_derivs(r, y, kappa, Gamma):
    P, m = y
    if P <= 0:
        return np.array([0, 0])
    
    rho0 = (P/kappa)**(1.0/Gamma)
    rho = rho0 + P/(Gamma - 1.0)
    dP_dr = -((rho + P) * (m + 4*np.pi*r**3*P)) / (r*(r - 2*m))
    dm_dr = 4*np.pi*r**2 * rho
    return np.array([dP_dr, dm_dr])

# Integrate to P=0
def integrate_tov(rho0_c, kappa, Gamma, h=1e-3, r_max=20):
    # Central pressure from the polytropic EoS
    P_c = kappa * rho0_c**Gamma
    rho_c = rho0_c + P_c/(Gamma - 1.0)
    r0 = 1e-3  
    m0 = (4*np.pi/3)*rho_c*r0**3

    # series expansion: P(r) ≃ P_c + P₂ r², P₂ = -½ (ρ_c+P_c)[(4π/3)ρ_c + 4πP_c]
    P2 = -0.5*(rho_c + P_c)*((4*np.pi/3)*rho_c + 4*np.pi*P_c)
    P0 = P_c + P2 * r0**2
    r = r0
    y = np.array([P0, m0])
    r_arr = [r]
    y_arr = [y.copy()]
    while y[0] > 0 and r < r_max:
        y = rk4_step(tov_derivs, r, y, h, args=(kappa, Gamma))
        r += h
        r_arr.append(r)
        y_arr.append(y.copy())
    # Linear interpolation between last two points to estimate r where P vanishes
    if len(r_arr) >= 2 and y_arr[-2][0] > 0 and y_arr[-1][0] <= 0:
        P_prev, m_prev = y_arr[-2]
        P_curr, m_curr = y_arr[-1]
        r_prev = r_arr[-2]
        r_curr = r_arr[-1]
        r_surf = r_prev + (r_curr - r_prev)*(-P_prev)/(P_curr - P_prev)
        m_surf = m_prev + (m_curr - m_prev)*(-P_prev)/(P_curr - P_prev)
    else:
        r_surf = r_arr[-1]
        m_surf = y_arr[-1][1]
    return r_surf, m_surf

# Wrapper and coversion geometric density unit ≃ 6.18e17 g/cm³
def make_star(rho0_c_cgs, kappa, Gamma):
    rho0_c = rho0_c_cgs / 6.18e17
    R, M = integrate_tov(rho0_c, kappa, Gamma)
    print(f"ρ₀,c = {rho0_c_cgs:.3e} g/cm³ -> R = {R:.4f}, M = {M:.4f}")
    return R, M

# ------------------------------------------------------------------------------------------------
# n                                 Basic star test                                         
# ------------------------------------------------------------------------------------------------


rho0_c_cgs = 5e14        
kappa = 3000              
Gamma = 2.5
R, M = make_star(rho0_c_cgs, kappa, Gamma)
# Expected: R ≃ 8.27, M ≃ 0.8657

# ----- Vary central density and plot M(ρ₀,c) -----
rho0_cgs_vals = np.logspace(14, 15, 50)
M_vals = []
for rho in rho0_cgs_vals:
    _, M_val = integrate_tov(rho/6.18e17, kappa, Gamma)
    M_vals.append(M_val)
plt.figure()
plt.semilogx(rho0_cgs_vals, M_vals, 'bo-')
plt.xlabel("Central Density ρ₀,c (g/cm³)")
plt.ylabel("Gravitational Mass M")
plt.title("M vs. Central Density")
plt.grid(True)
plt.tight_layout()
plt.show()
print("Maximum mass =", np.max(M_vals))

# -----  Vary κ and plot M(κ) -----
kappa_vals = [1000, 2000, 3000, 4000, 5000]
M_kappa = []
for k in kappa_vals:
    _, M_val = make_star(rho0_c_cgs, k, Gamma)
    M_kappa.append(M_val)
plt.figure()
plt.plot(kappa_vals, M_kappa, 'ro-')
plt.xlabel("κ")
plt.ylabel("Gravitational Mass M")
plt.title("M vs. κ")
plt.grid(True)
plt.tight_layout()
plt.show()

# -----  Vary Γ and plot M(Γ) -----
Gamma_vals = [2.0, 2.2, 2.5, 2.8, 3.0]
M_Gamma = []
for Gam in Gamma_vals:
    _, M_val = make_star(rho0_c_cgs, kappa, Gam)
    M_Gamma.append(M_val)
plt.figure()
plt.plot(Gamma_vals, M_Gamma, 'go-')
plt.xlabel("Γ")
plt.ylabel("Gravitational Mass M")
plt.title("M vs. Γ")
plt.grid(True)
plt.tight_layout()
plt.show()
