from ..data_misfit import L2DataMisfit
from ..fields import Fields
from ..utils import mkvc
import dask.array as da
from scipy.sparse import csr_matrix as csr
from dask import delayed
import time
import json
from threading import Thread, local
import socket
import select
import struct
import time
import sys
from .worker_utils.worker_communication import workerRequest
import numpy as np


def dias_deriv(self, m, f=None):
    """
    Distributed :obj:`simpeg.data_misfit.L2DataMisfit.deriv`
    """

    # create the request stream
    deriv_requests = {"request": 'deriv'}
    tc = time.time()
    
    # get predicted data from workers
    worker_threads = []
    results = [None] * len(self.simulation.cluster_worker_ids)
    cnt_host = 0

    for address in self.simulation.cluster_worker_ids:
        p = Thread(target=workerRequest, args=(results, deriv_requests, address, cnt_host))
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


L2DataMisfit.deriv = dias_deriv


def dias_deriv2(self, m, v, f=None):
    """
    Distributed :obj:`simpeg.data_misfit.L2DataMisfit.deriv2`
    """
    
    # create the request stream
    deriv2_requests = {"request": 'deriv2',
                       "vector": v.tolist(),
                       "model": m.tolist()}
    tc = time.time()
    
    # get predicted data from workers
    worker_threads = []
    results = [None] * len(self.simulation.cluster_worker_ids)
    cnt_host = 0
    
    for address in self.simulation.cluster_worker_ids:
        p = Thread(target=workerRequest, args=(results, deriv2_requests, address, cnt_host))
        p.start()
        worker_threads += [p]
        cnt_host += 1

    # join the threads to retrieve data
    for thread_ in worker_threads:
        print("joining .......................")
        thread_.join()
        print(f"[INFO] thread completed in: {time.time()-tc} sec")
    print("joining complete")
    # construct the predicted data vector
    data = np.sum(np.vstack(results), axis=0)

    return data


L2DataMisfit.deriv2 = dias_deriv2
