import numpy as np
import matplotlib.pyplot as plt

class ThermalBeam_th:
    def __init__(self, a, Ta, Lbda, h, q0):
        self.a = a               # Thickness of the beam [m]
        self.Ta = Ta             # Ambient temperature [°C]
        self.Lbda = Lbda         # Thermal conductivity [W/m/°C]
        self.h = h               # Convective loss coefficient [W/m^2/°C]
        self.q0 = q0             # Heat flow source [W/m^2]

    def solve(self, x):
        self.x = x
        self.L = x[-1]           # Total length of the beam [m]
        omega = np.sqrt(4 * self.h / (self.Lbda * self.a))

        # Temperature calculation
        num = self.Lbda * omega * np.cosh(omega * (self.L - self.x)) \
              + self.h * np.sinh(omega * (self.L - self.x))
        denom = self.Lbda * omega * np.sinh(omega * self.L) \
                + self.h * np.cosh(omega * self.L)
        self.T = self.Ta + self.q0 / (self.Lbda * omega) * num / denom

        # Heat flow calculation
        num = self.Lbda * omega * np.sinh(omega * (self.L - self.x)) \
              + self.h * np.cosh(omega * (self.L - self.x))
        self.q = self.q0 * num / denom

class Fig_temp:
    def __init__(self):
        # Initialization of a matplotlib figure
        cm = 1 / 2.54  # Convert cm to inches
        fig, ax = plt.subplots(2, 1, figsize=(16 * cm, 10 * cm))
        self.fig = fig
        self.ax = ax

        # Subplot for the temperature
        ax[0].set_title('Temperature', fontweight='bold')
        ax[0].set_xlabel('Position [m]')
        ax[0].set_ylabel('T(x) [°C]')
        ax[0].grid()
        ax[0].minorticks_on()
        ax[0].grid(True, which="minor", linestyle="--", linewidth=0.5, alpha=0.5)

        # Subplot for the heat flow
        ax[1].set_title('Heat flow', fontweight='bold')
        ax[1].set_xlabel('Position [m]')
        ax[1].set_ylabel('q(x) [W/m²]')
        ax[1].grid()
        ax[1].minorticks_on()
        ax[1].grid(True, which="minor", linestyle="--", linewidth=0.5, alpha=0.5)

        fig.tight_layout()  # Adjust the space between axes automatically

    def plot(self, obj, **kwargs):
        self.ax[0].plot(obj.x, obj.T, **kwargs)
        self.ax[1].plot(obj.x, obj.q, **kwargs)

    def legend(self, **kwargs):
        self.ax[0].legend(**kwargs)
        self.ax[1].legend(**kwargs)
