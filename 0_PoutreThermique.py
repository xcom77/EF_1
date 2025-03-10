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
fig.show()
