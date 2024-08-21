import casadi
import numpy as np

'''Numerical functions'''
def local_basis(q0, J_holonomic):
    J0 = J_holonomic(q0)
    nhc, ngc = J0.shape
    Q , _ = casadi.qr(J0.T)  # column of Q -> row space of J0 / normal space of manifold
    
    # null space of J0 -> tangent space of manifold
    I = casadi.DM.eye(ngc)
    k = 0
    basis = Q
    for i in range(ngc):
        vertI = I[:,i] - basis @ basis.T @ I[:,i]
        if casadi.norm_2(vertI) >= 10e-3:
            basis = casadi.horzcat(basis, vertI / casadi.norm_2(vertI))
            k += 1
        if k == ngc - nhc:
            break
    return basis  #[normal, tangent] = [u, v]

opts = {'abstol':1e-10,
        'max_iter':1000}

def q_local_to_manifold(q0, q0_basis, v, F_holonomic):  # q = q0 + V_basis @ v - U_basis @ h(v)          # ndof -> ngc
    ngc = q0.shape[0]
    ndof = v.shape[0]
    nhc = ngc-ndof
    h = casadi.SX.sym("h", nhc, 1)
    Feq = casadi.Function("Feq", [h], [F_holonomic(q0 + q0_basis[:,nhc:] @ v + q0_basis[:,:nhc] @ h)])
    h_v = casadi.rootfinder('Phi', 'newton', Feq, opts)
    return np.squeeze(q0 + q0_basis[:,nhc:] @ v + q0_basis[:,:nhc] @ h_v(np.zeros(nhc)))


'''Constrained manifold to local coordinates'''
'''Symbolic functions using CasADi'''

class LocalChart():   
    def __init__(self, ngc, nhc, ndof, J_holonomic):
        self.ngc = ngc
        self.nhc = nhc
        self.ndof = ndof
        self.q = casadi.SX.sym("q", ngc, 1)
        self.dq = casadi.SX.sym("dq", ngc, 1)
        self.q0 = casadi.SX.sym("q0", ngc, 1)
        self.q0_basis = casadi.SX.sym("q0_basis", ngc, ngc)
        self.Vbasis = self.q0_basis[:,nhc:]  # tangent space
        self.Ubasis = self.q0_basis[:,:nhc]  # normal space

        self.q_lc = casadi.SX.sym("q_lc", ndof, 1)
        self.dq_lc = casadi.SX.sym("dq_lc", ndof, 1)
        self.ddq_lc = casadi.SX.sym("dq_lc", ndof, 1)


        Bq = casadi.inv(J_holonomic(self.q) @ self.Ubasis)
        self.Dq = self.Vbasis - self.Ubasis @ Bq @ J_holonomic(self.q) @ self.Vbasis


    def q_manifold_to_local(self):    # ngc -> ndof
        return casadi.Function(
            "Phi", 
            [self.q, self.q0, self.q0_basis], 
            [self.Vbasis.T @ (self.q-self.q0)])  
    
    def dq_manifold_to_local(self):   # ngc -> ndof
        return casadi.Function(
            "dPhi", 
            [self.dq, self.q0_basis], 
            [self.Vbasis.T @ self.dq])       
    
    def dq_local_to_manifold(self):      # ndof -> ngc
        return casadi.Function(                                                        
            "dPhi_inv", 
            [self.dq_lc, self.q, self.q0_basis], 
            [self.Dq @ self.dq_lc])

    def ddq_local_to_manifold(self):     # ndof -> ngc
        return casadi.Function(                                                       
            "ddPhi_inv",
            [self.dq_lc, self.ddq_lc, self.q, self.dq, self.q0_basis],
            [self.Dq @ self.ddq_lc + casadi.jtimes(self.Dq, self.q, self.dq) @ self.dq_lc])   #- Ubasis @ Bq @ casadi.jtimes(J_holonomic(q), q, Dq@dq) @ dq_lc]
    
    def getFuncs(self):
        return self.q_manifold_to_local(), self.dq_manifold_to_local(), self.dq_local_to_manifold(), self.ddq_local_to_manifold()
    

    def J_Local(self):             # ->  2ndof * 2ngc
        return casadi.Function(     
            "J_Local", 
            [self.q0_basis], 
            [
                casadi.vertcat(casadi.horzcat(self.Vbasis.T, casadi.DM.zeros(self.ndof, self.ngc)),
                            casadi.horzcat(casadi.DM.zeros(self.ndof, self.ngc), self.Vbasis.T))
            ])


    def F_LocalAcc(self, F_ConstrAcc):
        u = casadi.SX.sym("u", self.ndof, 1)
        J = self.J_Local()
        return casadi.Function(
            "F_LocalAcc",
            [self.q, self.dq, u, self.q0_basis],
            [J(self.q0_basis) @ F_ConstrAcc(self.q, self.dq, u)])
