# @author: renaudf

import numpy as np
import FE_library as mef

# ==================================================
# Parameters of the thermal problem
# ==================================================
a = 0.005    # Thickness of the beam [m]
L = 0.1      # Total length of the beam [m]
Ta = 25      # Ambient temperature [°C]
Lbda = 60    # Thermal conductivity [W/m/°C]
h = 10       # Convective loss coefficient [W/m^2/°C]
q0 = 2000    # Heat flow source [W/m^2]

# ==================================================
# Analytical solution of the problem
# ==================================================
# Initialization of the analytical model
theo = mef.ThermalBeam_th(a, Ta, Lbda, h, q0)

# Discretization of the beam
x_theo = np.linspace(0, L, 100)

# Problem resolution
theo.solve(x_theo)

# Plot of the analytical solution of the temperature
fig = mef.Fig_temp()
fig.plot(theo, label="Analytical", color="k")
fig.legend()

# ==============================================================
# Solution with a single linear element
# ==============================================================
fe_1el_lin = mef.ThermalBeam_fe_1el_lin(a, Ta, Lbda, h, q0)
fe_1el_lin.solve(L)
fe_1el_lin.postprocess(x_theo)
fig.plot(fe_1el_lin, label="1 linear element", color="r")
fig.legend()

# ==================================================
# Solution with several linear elements
# ==================================================
fe_Nel_lin = mef.ThermalBeam_fe_Nel_lin(a, Ta, Lbda, h, q0)
# 4 elements and 5 nodes
fe_Nel_lin.mesh(L, 5)
fe_Nel_lin.solve()
fe_Nel_lin.postprocess()
fig.plot(fe_Nel_lin, label="4 Linear elem.", color="b",
         marker='o', markerfacecolor='none', markersize=3)
fig.legend(fontsize=8)
# 12 elements and 13 nodes
fe_Nel_lin.mesh(L, 13)
fe_Nel_lin.solve()
fe_Nel_lin.postprocess()
fig.plot(fe_Nel_lin, label="12 Linear elem.", color=[0, 0.8, 0],
         marker='o', markerfacecolor='none', markersize=3)
fig.legend(fontsize=8)

# ==================================================
# Solution with quadratic element and reference elements
# ==================================================
fe_quad = mef.ThermalBeam_fe_quad(a, Ta, Lbda, h, q0)
# Discretization of the beam
xn = np.linspace(0, L, 3)
fe_quad.mesh(xn)
fe_quad.solve()
fe_quad.postprocess()
fig.plot(fe_quad, label="2 quadratic elem.", color='k', linestyle='--', 
         marker='o', markerfacecolor='none', markersize=3)
fig.legend(fontsize=8)

fig.show()
