from .....electromagnetics.static.resistivity.simulation import BaseDCSimulation as Sim
from .....utils import Zero, mkvc
from .....data import Data
from ....utils import compute_chunk_sizes
from ....worker_utils.worker_communication import workerRequest
import dask
import dask.array as da
from dask.distributed import Future
import numpy as np
import zarr
import os
import shutil
import numcodecs
import json
from threading import Thread, local 
import socket
import select
import struct
import time
import sys

numcodecs.blosc.use_threads = False

Sim.sensitivity_path = './sensitivity/'
Sim.cluster_worker_ids = []


def dias_fields(self, m=None, return_Ainv=False):
    if m is not None:
        self.model = m


    A = self.getA()
    Ainv = self.solver(A, **self.solver_opts)
    RHS = self.getRHS()

    f = self.fieldsPair(self, shape=RHS.shape)
    f[:, self._solutionType] = Ainv * RHS

    Ainv.clean()

    if return_Ainv:
        return f, self.Solver(A.T, **self.solver_opts)
    else:
        return f, None


Sim.fields = dias_fields


def dias_getJtJdiag(self, m, W=None):
    """
        Return the diagonal of JtJ
    """
    self.model = m
    if self.gtgdiag is None:
        if isinstance(self.Jmatrix, Future):
            self.Jmatrix  # Wait to finish
        # Need to check if multiplying weights makes sense
        if W is None:
            self.gtgdiag = da.sum(self.Jmatrix ** 2, axis=0).compute()
        else:
            w = da.from_array(W.diagonal(), chunks='auto')[:, None]
            self.gtgdiag = da.sum((w * self.Jmatrix) ** 2, axis=0).compute()

    return self.gtgdiag


Sim.getJtJdiag = dias_getJtJdiag


def dias_deriv2_call(self, v):
    """
        Compute sensitivity matrix (J) and vector (v) product.
    """

    # create the request stream
    jvec_requests = {}
    jvec_requests["request"] = 'deriv2'
    jvec_requests["vector"] = v.tolist()
    tc = time.time()
    
    # get predicted data from workers
    worker_threads = []
    results = [None] * len(self.cluster_worker_ids)
    cnt_host = 0
    
    for address in self.cluster_worker_ids:
        p = Thread(target=workerRequest, args=(results, jvec_requests, address, cnt_host))
        p.start()
        worker_threads += [p]
        cnt_host += 1

    # join the threads to retrieve data
    for thread_ in worker_threads:
        print("joining .......................")
        thread_.join()
        print(f"[INFO] thread completed in: {time.time()-tc} sec")
    print("joining complete")
    # contruct the predicted data vector
    data = np.sum(np.vstack(results), axis=0)

    return data


Sim.dias_deriv2_call = dias_deriv2_call


def dias_Jvec(self, m, v, f=None):
        """
        Compute sensitivity matrix (J) and vector (v) product.
        """

        if self.store_sensitivities:
            J = self.getJ(m, f=f)
            return J.dot(v)

        self.model = m

        if self._mini_survey is not None:
            survey = self._mini_survey
        else:
            survey = self.survey

        Jv = []
        for source in survey.source_list:
            u_source = f[source, self._solutionType]  # solution vector
            dA_dm_v = self.getADeriv(u_source, v)
            dRHS_dm_v = self.getRHSDeriv(source, v)
            du_dm_v = self.Ainv * (-dA_dm_v + dRHS_dm_v)
            for rx in source.receiver_list:
                df_dmFun = getattr(f, "_{0!s}Deriv".format(rx.projField), None)
                df_dm_v = df_dmFun(source, du_dm_v, v, adjoint=False)
                Jv.append(rx.evalDeriv(source, self.mesh, f, df_dm_v))
        Jv = np.hstack(Jv)
        return self._mini_survey_data(Jv)


Sim.dias_Jvec = dias_Jvec


def dias_Jtvec_call(self):
    """
        Compute adjoint sensitivity matrix (J^T) and vector (v) product.
    """

    # create the request stream
    jtvec_requests = {}
    jtvec_requests["request"] = 'jtvec'
    tc = time.time()
    
    # get predicted data from workers
    worker_threads = []
    results = [None] * len(self.cluster_worker_ids)
    cnt_host = 0

    for address in self.cluster_worker_ids:
        p = Thread(target=workerRequest, args=(results, jtvec_requests, address, cnt_host))
        p.start()
        worker_threads += [p]
        cnt_host += 1

    # join the threads to retrieve data
    for thread_ in worker_threads:
        print("joining .......................")
        thread_.join()
        print(f"[INFO] thread completed in: {time.time()-tc} sec")

    # contruct the predicted data vector
    data = np.sum(np.vstack(results), axis=0)

    return data

Sim.Jtvec = dias_Jtvec_call


def dias_Jtvec(self, m, v, f=None):
        """
        Compute adjoint sensitivity matrix (J^T) and vector (v) product.
        """

        self.model = m

        if self.store_sensitivities:
            J = self.getJ(m, f=f)
            return np.asarray(J.T.dot(v))

        return self._Jtvec(m, v=v, f=f)


Sim.dias_Jtvec = dias_Jtvec


def compute_J(self, f=None, Ainv=None):

    if f is None:
        f, Ainv = self.fields(self.model, return_Ainv=True)

    m_size = self.model.size
    row_chunks = int(np.ceil(
        float(self.survey.nD) / np.ceil(float(m_size) * self.survey.nD * 8. * 1e-6 / self.max_chunk_size)
    ))
    Jmatrix = zarr.open(
        self.sensitivity_path + f"J.zarr",
        mode='w',
        shape=(self.survey.nD, m_size),
        chunks=(row_chunks, m_size)
    )

    blocks = []
    count = 0
    for source in self.survey.source_list:
        u_source = f[source, self._solutionType]

        for rx in source.receiver_list:

            PTv = rx.getP(self.mesh, rx.projGLoc(f)).toarray().T

            for dd in range(int(np.ceil(PTv.shape[1] / row_chunks))):
                start, end = dd*row_chunks, np.min([(dd+1)*row_chunks, PTv.shape[1]])
                df_duTFun = getattr(f, "_{0!s}Deriv".format(rx.projField), None)
                df_duT, df_dmT = df_duTFun(source, None, PTv[:, start:end], adjoint=True)
                ATinvdf_duT = Ainv * df_duT
                dA_dmT = self.getADeriv(u_source, ATinvdf_duT, adjoint=True)
                dRHS_dmT = self.getRHSDeriv(source, ATinvdf_duT, adjoint=True)
                du_dmT = -dA_dmT
                if not isinstance(dRHS_dmT, Zero):
                    du_dmT += dRHS_dmT
                if not isinstance(df_dmT, Zero):
                    du_dmT += df_dmT

                #
                du_dmT = du_dmT.T.reshape((-1, m_size))

                if len(blocks) == 0:
                    blocks = du_dmT
                else:
                    blocks = np.vstack([blocks, du_dmT])

                while blocks.shape[0] >= row_chunks:
                    Jmatrix.set_orthogonal_selection(
                        (np.arange(count, count + row_chunks), slice(None)),
                        blocks[:row_chunks, :].astype(np.float32)
                    )

                    blocks = blocks[row_chunks:, :].astype(np.float32)
                    count += row_chunks

                del df_duT, ATinvdf_duT, dA_dmT, dRHS_dmT, du_dmT

    if len(blocks) != 0:
        Jmatrix.set_orthogonal_selection(
            (np.arange(count, self.survey.nD), slice(None)),
            blocks.astype(np.float32)
        )

    del Jmatrix
    Ainv.clean()

    return da.from_zarr(self.sensitivity_path + f"J.zarr")


Sim.compute_J = compute_J


# This could technically be handled by dask.simulation, but doesn't seem to register
def dias_dpred_call(self, m=None, f=None, compute_J=False):
    """
    dpred(m, f=None)
    Create the projected data from a model.
    The fields, f, (if provided) will be used for the predicted data
    instead of recalculating the fields (which may be expensive!).

    .. math::

        d_\\text{pred} = P(f(m))

    Where P is a projection of the fields onto the data space.
    """
    if self.survey is None:
        raise AttributeError(
            "The survey has not yet been set and is required to compute "
            "data. Please set the survey for the simulation: "
            "simulation.survey = survey"
        )
    tc = time.time()
    # create the request stream
    dpred_requests = {}
    dpred_requests["request"] = 'dpred'
    dpred_requests["compute_j"] = False
    dpred_requests["model"] = m.tolist()

    # get predicted data from workers
    worker_threads = []
    results = [None] * len(self.cluster_worker_ids)
    cnt_host = 0

    # create a thread for each worker
    for address in self.cluster_worker_ids:
        p = Thread(target=workerRequest, args=(results, dpred_requests, address, cnt_host))
        p.start()
        worker_threads += [p]
        cnt_host += 1

    # join the threads to retrieve data
    for thread_ in worker_threads:
        print("joining .......................")
        thread_.join()
        print(f"[INFO] thread completed in: {time.time()-tc} sec")

    # contruct the predicted data vector
    data = np.hstack(results)

    return mkvc(data)


Sim.dpred = dias_dpred_call


def dias_dpred(self, m=None, f=None, compute_J=False):
        """
        dpred(m, f=None)
        Create the projected data from a model.
        The fields, f, (if provided) will be used for the predicted data
        instead of recalculating the fields (which may be expensive!).

        .. math::

            d_\\text{pred} = P(f(m))

        Where P is a projection of the fields onto the data space.
        """

        if self.survey is None:
            raise AttributeError(
                "The survey has not yet been set and is required to compute "
                "data. Please set the survey for the simulation: "
                "simulation.survey = survey"
            )

        if f is None:
            if m is None:
                m = self.model
            f, Ainv = self.fields(m, return_Ainv=True)
        self.Ainv = Ainv
        data = Data(self.survey)
        for src in self.survey.source_list:
            for rx in src.receiver_list:
                data[src, rx] = rx.eval(src, self.mesh, f)

        if compute_J:
            Jmatrix = self.compute_J(f=f, Ainv=Ainv)
            return (mkvc(data), Jmatrix)

        return mkvc(data), f

Sim.dias_dpred = dias_dpred


def dask_getSourceTerm(self):
    """
    Evaluates the sources, and puts them in matrix form
    :rtype: tuple
    :return: q (nC or nN, nSrc)
    """

    if getattr(self, "_q", None) is None:


        if self._mini_survey is not None:
            Srcs = self._mini_survey.source_list
        else:
            Srcs = self.survey.source_list

        if self._formulation == "EB":
            n = self.mesh.nN
            # return NotImplementedError

        elif self._formulation == "HJ":
            n = self.mesh.nC

        q = np.zeros((n, len(Srcs)), order="F")

        for i, source in enumerate(Srcs):
            q[:, i] = source.eval(self)

        self._q = q

    return self._q


Sim.getSourceTerm = dask_getSourceTerm

