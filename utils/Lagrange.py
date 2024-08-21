import casadi
import numpy as np
from numpy.polynomial.legendre import Legendre

def Normalize_LGL_Pts_Wts(n):
    """Lagrange-Gauss-Lobatto points and weights"""
    # Legendre polynomial of degree n-1
    P = Legendre.basis(n-1)
    # Derivative of the Legendre polynomial
    P_prime = P.deriv()
    # Legendre-Gauss-Lobatto points (roots of (1-x^2)*P'(x))
    Pts = np.concatenate(([-1], P_prime.roots(), [1]))
    # Weights calculation
    Wts = 2 / (n * (n-1) * (P(Pts)**2))
    return Pts, Wts

# Define Lagrange polynomials
def lagrange_basis(i, pts, t):
    li = 1
    for j in range(pts.shape[0]):
        if j != i:
            li *= (t - pts[j]) / (pts[i] - pts[j])
    return li

def differentiation_matrix(n, pts):  # degree: n-1
    x = casadi.SX.sym('x')
    # Coefficients of the collocation equation
    D1 = np.zeros((n,n))
    D2 = np.zeros((n,n))
    L = [lagrange_basis(i, pts, x) for i in range(n)]
    for j in range(n):  # j: basis index
        Lj = L[j]
        L1j = casadi.jacobian(Lj, x)
        L2j = casadi.jacobian(L1j, x)
        for r in range(n): # r: collocation point index
            #L[r, j] = casadi.Function('L', [x], [Lj])(pts[r])
            D1[r, j] = casadi.Function('L1j', [x], [L1j])(pts[r])
            D2[r, j] = casadi.Function('L2j', [x], [L2j])(pts[r])
    return D1, D2

def LagrangeInterpolation(t, collo_pts, collo_vals):
    L = [lagrange_basis(i, collo_pts, t) for i in range(collo_pts.shape[0])]
    return sum([ L[i] * collo_vals[i] for i in range(collo_pts.shape[0])])

def DiffLangrangeInterpolation(t, collo_pts, collo_vals):
    L = [lagrange_basis(i, collo_pts, t) for i in range(collo_pts.shape[0])]
    L_prime = [casadi.jacobian(L[i], t) for i in range(collo_pts.shape[0])]
    return sum([ L_prime[i] * collo_vals[i] for i in range(collo_pts.shape[0])])
