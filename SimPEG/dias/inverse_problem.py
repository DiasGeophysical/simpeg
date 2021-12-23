from ..inverse_problem import BaseInvProblem
from ..utils import Zero, mkvc
import numpy as np
import gc
from ..regularization import BaseComboRegularization, Sparse
from ..data_misfit import BaseDataMisfit
from ..objective_function import BaseObjectiveFunction
import time
from threading import Thread, local
import time
import sys
from .worker_utils.worker_communication import worker_request
import logging


def get_dpred(self, m, f=None, compute_J=False):
    """
    dpred(m, f=None)
    Create the projected data from a model.
    The fields, f, (if provided) will be used for the predicted data
    instead of recalculating the fields (which may be expensive!).

    .. math::

        d_\\text{pred} = P(f(m))

    Where P is a projection of the fields onto the data space.
    """

    tc = time.time()
    # create the request stream
    dpred_requests = {"request": 'dpred',
                      "compute_j": False,
                      "model": m.tolist()}

    # get predicted data from workers
    worker_threads = []
    results = [None] * len(self.dmisfit.simulation.cluster_worker_ids)
    cnt_host = 0

    # create a thread for each worker
    for address in self.dmisfit.simulation.cluster_worker_ids:
        p = Thread(target=worker_request, args=(results, dpred_requests, address, cnt_host))
        p.start()
        worker_threads += [p]
        cnt_host += 1

    # join the threads to retrieve data
    for thread_ in worker_threads:
        logging.info("joining predicted data call.......................")
        thread_.join()
        logging.info(f"[INFO] thread completed in: {time.time()-tc} sec")

    # construct the predicted data vector
    data = np.hstack(results)

    return mkvc(data)


BaseInvProblem.get_dpred = get_dpred


def dias_evalFunction(self, m, return_g=True, return_H=True):
    """

        evalFunction(m, return_g=True, return_H=True)

    """

    self.model = m
    gc.collect()

    # call dpred on the worker machine and grab the residual too (this will be a list)
    logging.info("get residuals..")
    phi_deez = self.get_dpred(m, compute_J=return_H)

    phi_d = 0
    for phi_sub_d in phi_deez:
        phi_d += phi_sub_d

    phi_d = np.asarray(phi_d)
    self.reg2Deriv = self.reg.deriv2(m)

    phi_m = self.reg(m)

    self.phi_d, self.phi_d_last = phi_d, self.phi_d
    self.phi_m, self.phi_m_last = phi_m, self.phi_m

    phi = phi_d + self.beta * phi_m
    out = (phi,)

    if return_g:

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
