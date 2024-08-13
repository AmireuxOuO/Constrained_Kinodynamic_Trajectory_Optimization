import casadi
from utils.Lagrange import *
import numpy as np

def VarDiff(Coeffs, Var):
    Res = casadi.SX(0)
    for i in range(len(Var)):
        Res += Coeffs[i] * Var[i]
    return Res

class LGL_collocation():
    def __init__(self, Ntp, ngc, ndof):
        self.Ntp = Ntp
        self.ngc = ngc
        self.ndof = ndof
        self.deltaT = casadi.SX.sym("t")                                        # Time interval
        self.Coffs = casadi.SX.sym('Coff', Ntp)

    def StatesDiff(self):
        cx_list = [casadi.SX.sym(f'x{i}', 2*self.ngc, 1) for i in range(self.Ntp)]   # generalized coordinates
        StatesDiff = casadi.Function(
            "Xdot",
            [casadi.vertcat(*cx_list), self.deltaT, self.Coffs],
            [
                2*VarDiff(self.Coffs, cx_list)/self.deltaT,
            ],
        )
        return StatesDiff

    def StateDiff(self):
        cv_list = [casadi.SX.sym(f'v{i}', self.ngc, 1) for i in range(self.Ntp)]
        StateDiff = casadi.Function(
            "qdot",
            [casadi.vertcat(*cv_list), self.deltaT, self.Coffs],
            [
                2*VarDiff(self.Coffs, cv_list)/self.deltaT
            ]
        )
        return StateDiff

    def LocalsDiff(self):
        cy_list = [casadi.SX.sym(f'y{i}', 2*self.ndof, 1) for i in range(self.Ntp)]  # local coordinates
        LocalsDiff = casadi.Function(
            "Ydot",
            [casadi.vertcat(*cy_list), self.deltaT, self.Coffs],
            [
                2*VarDiff(self.Coffs, cy_list)/self.deltaT
            ],
        )
        return LocalsDiff

    def LocalDiff(self):
        cp_list = [casadi.SX.sym(f'p{i}', self.ndof, 1) for i in range(self.Ntp)] 
        LocalDiff = casadi.Function(
            "pdot",
            [casadi.vertcat(*cp_list), self.deltaT, self.Coffs],
            [
                2*VarDiff(self.Coffs, cp_list)/self.deltaT
            ]
        )
        return LocalDiff
    
    def getDiffFuncs(self):
        return self.StatesDiff(), self.StateDiff(), self.LocalsDiff(), self.LocalDiff()