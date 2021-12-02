from ..data_misfit import L2DataMisfit
from ..fields import Fields
from ..utils import mkvc
import dask.array as da
from scipy.sparse import csr_matrix as csr
from dask import delayed


def dias_call(self, m, f=None):
    """
    Distributed :obj:`simpeg.data_misfit.L2DataMisfit.__call__`
    """
    R = self.W * self.residual(m, f=f)
    phi_d = 0.5 * da.dot(R, R)
    return self.client.compute(phi_d, workers=self.workers)


L2DataMisfit.__call__ = dias_call


def dias_deriv(self, m, f=None):
    """
    Distributed :obj:`simpeg.data_misfit.L2DataMisfit.deriv`
    """

    Jtvec = self.simulation.Jtvec()


    return Jtvec


L2DataMisfit.deriv = dias_deriv


def dias_deriv2(self, m, v, f=None):
    """
    Distributed :obj:`simpeg.data_misfit.L2DataMisfit.deriv2`
    """

    return self.simulation.dias_deriv2_call(v)


L2DataMisfit.deriv2 = dias_deriv2
