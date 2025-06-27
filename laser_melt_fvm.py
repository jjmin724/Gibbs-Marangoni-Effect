
"""
laser_melt_fvm.py

Minimal 2‑D finite‑volume implementation (CPU) of the
“Role of Marangoni Convection in a Repetitive Laser Melting Process”
model (Das & Kar, 2020).

• Collocated grid (u,v,T,fL,H at cell centres)
• Incompressible Navier‑Stokes + Enthalpy‑Porosity
• SIMPLE pressure‑velocity coupling solved by TDMA (line‑by‑line)
• Explicit first‑order time advancement
• Gaussian pulse laser source, Marangoni top shear, buoyancy
• All NumPy, hence Colab/CPU ready
--------------------------------------------------------------------
This script is a teaching / template code – cell numbers, material
properties and solver tolerance are set for demonstration and can be
modified for production runs.
"""

import numpy as np
import matplotlib.pyplot as plt

# ================================================================
# 1. --- domain & numerical parameters ---------------------------
# ----------------------------------------------------------------
Lx = 4.0e-3   # 4 mm half‑width (symmetry plane at x=0)
Ly = 2.0e-3   # 2 mm depth (0=bottom, Ly=free surface)
Nx = 120
Ny = 60
dx = Lx / Nx
dy = Ly / Ny
dt = 5e-6                # s, stable for explicit advection
n_steps = 4000           # ~20 ms simulation

# ---------------------------------------------------------------
# 2. --- material / laser / mushy zone constants ----------------
rho   = 7800.0           # kg/m3
mu    = 6.0e-3           # Pa·s (steel melt at ~1800 K)
k     = 25.0             # W/m/K (solid) ; assume same in liquid
Cp    = 700.0            # J/kg/K
L_lat = 2.5e5            # J/kg latent heat
Tmelt = 1700.0           # K
deltaT = 10.0            # mushy half‑width
sigma_T = -0.43e-3       # dσ/dT, N/m/K (negative)
beta  = 1.1e-4           # 1/K, thermal expansion
g     = 9.81
# mushy zone damping coefficients
C_mush = 1e6
B_mush = 1e-3

# laser
P_L   = 100.0            # W, nominal
eta   = 0.35             # absorptivity
P_a   = eta * P_L        # absorbed power
R_beam = 0.2e-3          # 0.2 mm beam radius (1/e^2)

pulse_period = 1e-3      # 1 kHz pulse period
pulse_duty   = 0.5       # 50 % duty
# environment
T_inf = 300.0            # K ambient
h_conv = 80.0            # W/m2/K
eps_surf = 0.2           # emissivity
sigma_SB = 5.670374419e-8

# ===============================================================
# Allocate fields (ghost layers included)
u = np.zeros((Ny+2, Nx+2))   # x‑velocity
v = np.zeros_like(u)         # y‑velocity
p = np.zeros_like(u)         # pressure
T = 300.0*np.ones_like(u)    # temperature
H = Cp*T                     # enthalpy
fL = np.zeros_like(u)        # liquid fraction

# ==============================================================
# 3. --- helper: TDMA solver -----------------------------------
def tdma(a,b,c,d):
    """Tridiagonal matrix algorithm"""
    n = len(d)
    cp = np.zeros(n)
    dp = np.zeros(n)
    x  = np.zeros(n)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1,n):
        denom = b[i] - a[i]*cp[i-1]
        cp[i] = c[i]/denom if i < n-1 else 0.0
        dp[i] = (d[i] - a[i]*dp[i-1])/denom
    x[-1] = dp[-1]
    for i in range(n-2,-1,-1):
        x[i] = dp[i] - cp[i]*x[i+1]
    return x

# =============================================================
# 4. --- boundary utilities -----------------------------------
def apply_temperature_bc(T):
    # adiabatic on left, right, bottom
    T[:,0]  = T[:,1]
    T[:,-1] = T[:,-2]
    T[0,:]  = T[1,:]      # bottom
    # top: convection + radiation treated via ghost cell
    q_conv = h_conv*(T[Ny,:]-T_inf) + eps_surf*sigma_SB*(T[Ny,:]**4 - T_inf**4)
    T[Ny+1,:] = T[Ny,:] + q_conv*dy/k
    return T

def apply_velocity_bc(u,v,T):
    # no‑slip on walls and bottom
    u[:,0] = 0.0; u[:,-1] = 0.0
    v[:,0] = 0.0; v[:,-1] = 0.0
    u[0,:] = 0.0; v[0,:] = 0.0
    # top free surface
    v[Ny+1,:] = 0.0
    # Marangoni shear
    dTdx = (T[Ny,2:] - T[Ny,:-2])/(2*dx)
    shear = sigma_T * dTdx
    u[Ny+1,1:-1] = u[Ny,1:-1] + shear*dy/mu
    u[Ny+1,0] = u[Ny+1,1]; u[Ny+1,-1]=u[Ny+1,-2]
    return u,v

# =============================================================
# 5. --- laser source term ------------------------------------
def laser_source(X,Y,time):
    phase = (time % pulse_period) / pulse_period
    on = 1.0 if phase < pulse_duty else 0.0
    if on == 0.0:
        return np.zeros_like(X)
    q = (2*P_a)/(np.pi*R_beam**2) * np.exp(-2*(X**2)/R_beam**2)
    return q

# pre‑compute cell centre coordinates
xc = (np.arange(Nx)+0.5)*dx - Lx/2
yc = (np.arange(Ny)+0.5)*dy
Xc,Yc = np.meshgrid(xc,yc)

# =============================================================
# 6. --- main time loop ---------------------------------------
for step in range(1, n_steps+1):
    time = step*dt

    # --- 6.1 liquid fraction & enthalpy -----------------------
    fL = np.clip((T - (Tmelt-deltaT)) / (2*deltaT), 0.0, 1.0)
    H  = Cp*T + fL*L_lat
    A_mush = C_mush * (1-fL)**2 / (fL**3 + B_mush)

    # --- 6.2 momentum equations (explicit upwind) ------------
    u_old = u.copy(); v_old = v.copy()
    for j in range(1,Ny+1):
        for i in range(1,Nx+1):
            ue = 0.5*(u_old[j,i]+u_old[j,i+1])
            uw = 0.5*(u_old[j,i]+u_old[j,i-1])
            vn = 0.5*(v_old[j,i]+v_old[j+1,i])
            vs = 0.5*(v_old[j,i]+v_old[j-1,i])
            du2dx = (ue**2 - uw**2)/dx
            duvdy = (u_old[j+1,i]*vn - u_old[j-1,i]*vs)/(2*dy)
            lap_u = (u_old[j,i+1]-2*u_old[j,i]+u_old[j,i-1])/dx**2 +                     (u_old[j+1,i]-2*u_old[j,i]+u_old[j-1,i])/dy**2

            dv2dy = (vn**2 - vs**2)/dy
            duvdx = (u_old[j,i+1]*ue - u_old[j,i-1]*uw)/(2*dx)
            lap_v = (v_old[j,i+1]-2*v_old[j,i]+v_old[j,i-1])/dx**2 +                     (v_old[j+1,i]-2*v_old[j,i]+v_old[j-1,i])/dy**2

            u[j,i] = u_old[j,i] + dt*( -du2dx - duvdy                        + mu/rho*lap_u - A_mush[j,i]/rho*u_old[j,i] )
            buoy = beta*g*(T[j,i]-T_inf)
            v[j,i] = v_old[j,i] + dt*( -duvdx - dv2dy                        + mu/rho*lap_v - A_mush[j,i]/rho*v_old[j,i] + buoy )

    # --- 6.3 pressure correction (Jacobi Poisson) ------------
    div = (u[1:-1,2:] - u[1:-1,1:-1])/dx + (v[2:,1:-1] - v[1:-1,1:-1])/dy
    rhs = rho/dt * div
    for _ in range(30):
        p_old = p.copy()
        p[1:-1,1:-1] = 0.25*( p[1:-1,2:] + p[1:-1,:-2] +                               p[2:,1:-1] + p[:-2,1:-1] - dx*dy*rhs )
        p[:,-1] = p[:,-2]; p[:,0]=p[:,1]; p[0,:]=p[1,:]; p[-1,:]=p[-2,:]
        if np.max(np.abs(p - p_old)) < 1e-3:
            break
    u[1:-1,1:-1] -= dt/rho * (p[1:-1,1:-1] - p[1:-1,:-2]) / dx
    v[1:-1,1:-1] -= dt/rho * (p[1:-1,1:-1] - p[:-2,1:-1]) / dy

    # --- 6.4 energy equation (implicit TDMA along x) ----------
    q_ls = laser_source(Xc, Yc, time)
    alpha = k/(rho*Cp)
    for j in range(Ny):
        a = -alpha*dt/dx**2*np.ones(Nx)
        b = 1+2*alpha*dt/dx**2*np.ones(Nx)
        c = -alpha*dt/dx**2*np.ones(Nx)
        d = T[j+1,1:-1] + dt*( - ( (u[j+1,1:-1] *                                     (T[j+1,1:-1]-T[j+1,:-2]) / dx) +                                     (v[j+1,1:-1] *                                     (T[j+1,1:-1]-T[j,1:-1]) / dy) ) +                                     q_ls[j,:] / (rho*Cp) )
        Trow = tdma(a, b, c, d)
        T[j+1,1:-1] = Trow
    T = apply_temperature_bc(T)

    # --- 6.5 velocity boundaries after T update --------------
    u, v = apply_velocity_bc(u, v, T)

    # --- 6.6 progress output ---------------------------------
    if step % 1000 == 0:
        print(f"step {step}/{n_steps}, t={time*1e3:.2f} ms, maxT={T.max():.1f} K")

# ==============================================================
# Plot temperature field
plt.figure(figsize=(5,4))
plt.contourf(Xc*1e3, Yc*1e3, T[1:-1,1:-1], 30)
plt.colorbar(label='T [K]')
plt.xlabel('x [mm]'); plt.ylabel('y [mm]')
plt.title('Temperature at final time')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
