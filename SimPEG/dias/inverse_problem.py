from ..inverse_problem import BaseInvProblem
import numpy as np

from dask import delayed
from dask.distributed import Future, get_client
import dask.array as da
import gc
from ..regularization import BaseComboRegularization, Sparse
from ..data_misfit import BaseDataMisfit
from ..objective_function import BaseObjectiveFunction


def dias_evalFunction(self, m, return_g=True, return_H=True):
    """

        evalFunction(m, return_g=True, return_H=True)

    """

    self.model = m
    gc.collect()

    # call dpred on the worker machine and grab the residual too (this will be a list)  
    print("got residuals..")
    phi_deez = self.dmisfit.simulation.dpred(m, compute_J=return_H)

    phi_d = 0
    for phi_sub_d in phi_deez:
        phi_d += phi_sub_d
    print("calculated phi_d..")
    phi_d = np.asarray(phi_d)
    # print(self.dpred[0])
    self.reg2Deriv = self.reg.deriv2(m)
    # reg = np.linalg.norm(self.reg2Deriv * self.reg._delta_m(m))
    phi_m = self.reg(m)

    self.phi_d, self.phi_d_last = phi_d, self.phi_d
    self.phi_m, self.phi_m_last = phi_m, self.phi_m

    phi = phi_d + self.beta * phi_m

    out = (phi,)
    if return_g:
        print("getting deriv..")
        phi_dDeriv = self.dmisfit.deriv(m, f=None)
        if hasattr(self.reg.objfcts[0], "space") and self.reg.objfcts[0].space == "spherical":
            phi_mDeriv = self.reg2Deriv * self.reg._delta_m(m)
        else:
            phi_mDeriv = self.reg.deriv(m)

        g = np.asarray(phi_dDeriv) + self.beta * phi_mDeriv
        out += (g,)

    if return_H:

        def H_fun(v):
            phi_d2Deriv = self.dmisfit.deriv2(m, v)
            if hasattr(self.reg.objfcts[0], "space") and self.reg.objfcts[0].space == "spherical":
                phi_m2Deriv = self.reg2Deriv * v
            else:
                phi_m2Deriv = self.reg.deriv2(m, v=v)

            H = phi_d2Deriv + self.beta * phi_m2Deriv

            return H

        H = H_fun
        out += (H,)
    return out if len(out) > 1 else out[0]


BaseInvProblem.evalFunction = dias_evalFunction
