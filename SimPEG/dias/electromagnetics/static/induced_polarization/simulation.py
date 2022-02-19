import numpy as np
from .....data import Data

from .....electromagnetics.static.induced_polarization.simulation import BaseIPSimulation as Sim
from .....utils import Zero, mkvc


def dias_fields(self, m=None, return_Ainv=False):

    if m is not None:
        self.model = m
        # sensitivity matrix is fixed
        # self._Jmatrix = None

    # if self._f is None:
    f = self.fieldsPair(self)

    A = self.getA()

    self.Ainv = self.solver(A.T, **self.solver_opts)

    rhs = self.getRHS()

    srcs = self.survey.source_list

    f[srcs, self._solutionType] = self.Ainv * rhs

    self._f = f

    if self._scale is None:

        scale = Data(self.survey, np.full(self.survey.nD, self._sign))
        # loop through recievers to check if they need to set the _dc_voltage
        for src in self.survey.source_list:
            for rx in src.receiver_list:
                if (
                        rx.data_type == "apparent_chargeability"
                        or self._data_type == "apparent_chargeability"
                ):
                    scale[src, rx] = self._sign / rx.eval(src, self.mesh, self._f)
        self._scale = scale.dobs


    # if not self.storeJ:
    #     self.Ainv.clean()

    self._pred = self.forward(m, f=self._f)

    if return_Ainv:
        return self._f, self.solver(A.T, **self.solver_opts)
    else:
        return self._f, None


Sim.fields = dias_fields


def dias_dpred(self, m=None, f=None):
    """

    Predicted data.

    .. math::

        d_\\text{pred} = Pf(m)

    """
    if f is None:

        f, Ainv = self.fields(m, return_Ainv=True)

        self.Ainv = Ainv

    return self._pred, f


Sim.dpred = dias_dpred
