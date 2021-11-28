from ..objective_function import ComboObjectiveFunction, BaseObjectiveFunction
import dask
import dask.array as da
import os
import shutil
import numpy as np
from dask.distributed import Future, get_client, Client


def dask_call(self, m, f=None):
    fcts = []
    multipliers = []
    for i, phi in enumerate(self):
        multiplier, objfct = phi
        if multiplier == 0.0:  # don't evaluate the fct
            continue
        else:

            if f is not None and objfct._has_fields:
                fct = objfct(m, f=f[i])
            else:
                fct = objfct(m)

            if isinstance(fct, Future):
                future = self.client.compute(
                    self.client.submit(da.multiply, multiplier, fct).result()
                )
                fcts += [future]
            else:
                fcts += [fct]

            multipliers += [multiplier]

    if isinstance(fcts[0], Future):
        phi = self.client.submit(
            da.sum, self.client.submit(da.vstack, fcts), axis=0
        ).result()
        return phi
    else:
        return np.sum(
            np.r_[multipliers][:, None] * np.vstack(fcts), axis=0
        ).squeeze()


ComboObjectiveFunction.__call__ = dask_call


def dias_deriv(self, residual):
    """
    First derivative of the composite objective function is the sum of the
    derivatives of each objective function in the list, weighted by their
    respective multplier.

    :param numpy.ndarray m: model
    :param SimPEG.Fields f: Fields object (if applicable)
    """

    g = []
    multipliers = []
    for i, phi in enumerate(self):
        multiplier, objfct = phi
        if multiplier == 0.0:  # don't evaluate the fct
            continue
        else:

            fct = objfct.deriv(residual)
            
            fct = da.multiply(multiplier, fct)
            
            g += [fct]

            multipliers += [multiplier]

    return np.sum(
        np.r_[multipliers][:, None] * np.vstack(g), axis=0
    ).squeeze()


ComboObjectiveFunction.deriv = dias_deriv


def dias_deriv2(self, v=None):
    """
    Second derivative of the composite objective function is the sum of the
    second derivatives of each objective function in the list, weighted by
    their respective multplier.

    :param numpy.ndarray m: model
    :param numpy.ndarray v: vector we are multiplying by
    :param SimPEG.Fields f: Fields object (if applicable)
    """

    H = []
    multipliers = []
    for i, phi in enumerate(self):
        multiplier, objfct = phi
        if multiplier == 0.0:  # don't evaluate the fct
            continue
        else:
            fct = objfct.deriv2(v)

            fct = da.multiply(multiplier, fct)
            H += [fct]

            multipliers += [multiplier]

    phi_deriv2 = 0
    for multiplier, h in zip(multipliers, H):
        phi_deriv2 += multiplier * h

    return phi_deriv2


ComboObjectiveFunction.deriv2 = dias_deriv2
