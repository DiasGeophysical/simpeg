from ..data_misfit import L2DataMisfit
from threading import Thread, local
import time
from .worker_utils.worker_communication import worker_request
import numpy as np
import logging


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
        p = Thread(target=worker_request, args=(results, deriv_requests, address, cnt_host))
        p.start()
        worker_threads += [p]
        cnt_host += 1

    # join the threads to retrieve data
    for thread_ in worker_threads:
        logging.info("joining 1st derivative.......................")
        thread_.join()
        logging.info(f"thread completed in: {time.time()-tc} sec")

    # construct the predicted data vector
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
        p = Thread(target=worker_request, args=(results, deriv2_requests, address, cnt_host))
        p.start()
        worker_threads += [p]
        cnt_host += 1

    # join the threads to retrieve data
    for thread_ in worker_threads:
        logging.info("joining 2nd derivative.......................")
        thread_.join()
        logging.info(f"thread completed in: {time.time()-tc} sec")
    logging.info("joining complete")
    # construct the predicted data vector
    data = np.sum(np.vstack(results), axis=0)

    return data


L2DataMisfit.deriv2 = dias_deriv2
