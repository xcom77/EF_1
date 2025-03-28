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

    def show(self, **kwargs):
        plt.show()

class ThermalBeam_fe_1el_lin:
    def __init__(self, a, Ta, Lbda, h, q0):
        self.a = a       # Thickness of the beam [m]
        self.Ta = Ta     # Ambient temperature [°C]
        self.Lbda = Lbda # Thermal conductivity [W/m/°C]
        self.h = h       # Convective loss coefficient [W/m^2/°C]
        self.q0 = q0     # Heat flow source [W/m^2]

    def solve(self, L):
        self.L = L
        # Matrices
        M1 = self.Lbda / self.L * np.array([[1, -1], [-1, 1]])
        M2 = self.h * np.array([[0, 0], [0, 1]])
        M3 = 4/6 * self.h * self.L / self.a * np.array([[2, 1], [1, 2]])

        # Vectors
        V1 = 4/2 * self.h * self.L * self.Ta / self.a * np.array([1, 1])
        V2 = self.q0 * np.array([1, 0])
        V3 = self.h * self.Ta * np.array([0, 1])

        # Resolution
        self.fe_M = M1 + M2 + M3
        self.fe_V = V1 + V2 + V3
        self.fe_Tn = np.linalg.solve(self.fe_M, self.fe_V)

    def postprocess(self, x):
        self.x = x
        # Polynomial basis evaluated at x
        P = np.ones((len(x), 2))
        P[:, 1] = x

        # Temperature interpolation
        L = self.L
        self.T = P @ (np.array([[1, 0], [-1/L, 1/L]]) @ self.fe_Tn)

        # Derivative of the polynomial basis evaluated at x
        P_x = np.zeros((len(x), 2))
        P_x[:, 1] = 1
        self.q = -self.Lbda * P_x @ (np.array([[1, 0], [-1/L, 1/L]]) @ self.fe_Tn)

class ThermalBeam_fe_Nel_lin:
    def __init__(self, a, Ta, Lbda, h, q0):
        self.a = a       # Thickness of the beam [m]
        self.Ta = Ta     # Ambient temperature [°C]
        self.Lbda = Lbda # Thermal conductivity [W/m/°C]
        self.h = h       # Convective loss coefficient [W/m^2/°C]
        self.q0 = q0     # Heat flow source [W/m^2]

    def mesh(self, L, N):
        self.L = L
        self.xn = np.linspace(0, L, N)
        self.le = L / (N-1)

    def solve(self):
        M1 = self.Lbda / self.le * np.array([[1, -1], [-1, 1]])
        M3 = 4/6 * self.h * self.le / self.a * np.array([[2, 1], [1, 2]])
        V1 = 4/2 * self.h * self.le * self.Ta / self.a * np.array([1, 1])
        Nb_nodes = len(self.xn)
        Nb_elem = Nb_nodes - 1
        self.fe_M = np.zeros((Nb_nodes, Nb_nodes))
        self.fe_V = np.zeros(Nb_nodes)
        for ind in range(0, Nb_elem):
            indices = ind + np.array([0, 1])
            self.fe_M[np.ix_(indices, indices)] += M1 + M3
            self.fe_V[np.ix_(indices)] += V1
        # 2nd term ==> h TL* TL
        self.fe_M[-1, -1] += self.h
        # 5th term ==> T0* q0
        self.fe_V[0] += self.q0
        # 6th term ==> h TL* TA
        self.fe_V[-1] += self.h * self.Ta
        # Resolution
        self.fe_Tn = np.linalg.solve(self.fe_M, self.fe_V)

    def postprocess(self):
        # Vector initialization
        Nb_nodes = len(self.xn)
        Nb_elem = Nb_nodes - 1
        self.x = np.zeros(2*Nb_elem)
        self.T = np.zeros(2*Nb_elem)
        self.q = np.zeros(2*Nb_elem)
        # Postprocessing of the temperature and heat flow
        for ind in range(0, Nb_elem):
            indices = ind + np.array([0, 1])
            ind_stock = 2*ind + np.array([0, 1])
            self.x[ind_stock] = self.xn[indices]
            self.T[ind_stock] = self.fe_Tn[indices]
            self.q[ind_stock] = - self.Lbda * np.ones(2) * np.diff(self.fe_Tn[indices]) / self.le
