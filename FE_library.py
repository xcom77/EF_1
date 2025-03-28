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

class Pt_Gauss:
    def __init__(self, N):
        self.N = N

        # Up to polynomial of order 1
        if N == 1:
            self.xi = np.array([0])
            self.w = np.array([2])

        # Up to polynomial of order 3
        elif N == 2:
            self.xi = 1 / np.sqrt(3) * np.array([-1, 1])
            self.w = np.ones(2)

        # Up to polynomial of order 5
        elif N == 3:
            self.xi = np.sqrt(3 / 5) * np.array([-1, 0, 1])
            self.w = np.array([5, 8, 5]) / 9
            
class Element_quad:
    def __init__(self, x, dof):
        # Here, x defines the nodes on the boundaries of the quadratic element
        # The middle node is added here
        self.x_el = np.array([x[0], (x[0] + x[1]) / 2, x[1]])
        self.le = x[1] - x[0]
        # Indices of the degree of freedom (dof) in the global matrices
        self.dof = dof
        # Jacobian J = dx / dξ
        self.J = self.le / 2
        # j = dξ / dx
        self.j = 2 / self.le
        Pn = np.array([
            [1, -1, 1],
            [1, 0, 0],
            [1, 1, 1]
        ])
        self.inv_Pn = np.linalg.inv(Pn)

    # Polynomial basis
    def P(self, xi):
        return np.array([ 1, xi, xi**2])

    # Derivative of the polynomial basis
    def P_xi(self, xi):
        return np.array([ 0, 1, 2*xi])

    # Interpolation function
    def N(self, xi):
        return self.P * (1/2) * self.inv_Pn

    # Derivative of the interpolation function
    def N_xi(self, xi):
        return self.P_xi * (1/2) * self.inv_Pn

    # Derivative of the interpolation function
    def N_x(self, xi):
        return self.j * self.N_xi

    # Computation of the elementary matrix M1
    def M1_el(self, coeff):
        # Gauss point
        Pt = Pt_Gauss(2)
        mat = np.zeros((3, 3))
        for ind in range(0, Pt.N):
            w = Pt.w[ind]  # Weight
            xi = Pt.xi[ind]  # Integration point (Gauss)
            N_x = self.N_x(xi)
            mat += w * np.outer(N_x, N_x) * self.J
        # Elementary matrix M1
        return coeff * mat

    # Computation of the elementary matrix M3
    def M3_el(self, coeff):
        # Gauss point
        Pt = Pt_Gauss(3)
        mat = np.zeros((3, 3))
        for ind in range(0, Pt.N):
            w = Pt.w[ind]  # Weight
            xi = Pt.xi[ind]  # Integration point (Gauss)
            N = self.N(xi)
            mat += w * np.outer(N, N) * self.J
        # Elementary matrix M3
        return coeff * mat

    # Computation of the elementary vector V4
    def V4_el(self, coeff):
        # Gauss point
        Pt = Pt_Gauss(2)
        vect = np.zeros(3)
        for ind in range(0, Pt.N):
            w = Pt.w[ind]  # Weight
            xi = Pt.xi[ind]  # Integration point (Gauss)
            N = self.N(xi)
            vect += w * N * self.J
        # Elementary matrix V4
        return coeff * vect

class ThermalBeam_fe_quad:
    def __init__(self, a, Ta, Lbda, h, q0):
        self.a = a  # Thickness of the beam [m]
        self.Ta = Ta  # Ambient temperature [°C]
        self.Lbda = Lbda  # Thermal conductivity [W/m°C]
        self.h = h  # Convective loss coefficient [W/m²°C]
        self.q0 = q0  # Heat flow source [W/m²]

    def mesh(self, xn):
        self.L = xn[-1]
        self.xn = xn
        # Number of nodes and elements
        nb_nodes = len(xn)
        nb_elem = nb_nodes - 1
        # Loop over the elements
        self.elem = []
        for ind in range(0, nb_elem):
            # Indices of the nodes (only boundary ones)
            ind_xn = ind + np.array([0, 1])
            x_el = self.xn[ind_xn]
            # Indices of the degree of freedom of the element (3 dof per element)
            dof = 2*ind + np.array([0, 1, 2])
            # Creation of an element
            self.elem.append(Element_quad(x_el, dof))

    def solve(self):
        # Matrix initialization
        Nb_dof = len(self.elem) + len(self.xn)
        self.fe_M = np.zeros((Nb_dof, Nb_dof))
        self.fe_V = np.zeros(Nb_dof)
        coeff_1 = self.Lbda
        coeff_3 = 4 * self.h / self.a
        coeff_4 = 4 * self.h / self.a * self.Ta
        # Matrix assembly for each element
        for el in self.elem:
            indices = el.dof
            ix_indices = np.ix_(indices, indices)
            self.fe_M[ix_indices] += el.M1_el(coeff_1) + el.M3_el(coeff_3)
            self.fe_V[ix_indices] += el.V4_el(coeff_4) 
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
        Nb_nodes = 3 * len(self.elem)
        self.x = np.zeros(Nb_nodes)
        self.T = np.zeros(Nb_nodes)
        self.q = np.zeros(Nb_nodes)
        # Postprocessing of the temperature and heat flow
        for ind, el in enumerate(self.elem):
            # Temperature value at the nodes of the current element
            dof = el.dof
            Tn_el = self.fe_Tn[dof]
            # Indices for storage
            ind_stock = 3*ind + np.array([0, 1, 2])
            # Position
            self.x[ind_stock] = el.x_el
            # Temperature
            self.T[ind_stock] = np.row_stack((el.N(-1), el.N(0), el.N(1))) @ Tn_el
            self.q[ind_stock] = - self.Lbda * np.row_stack((el.N_x(-1), el.N_x(0), el.N_x(1))) @ Tn_el

