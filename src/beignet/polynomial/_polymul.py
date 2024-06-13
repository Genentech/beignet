import numpy

from ._as_series import as_series
from ._trimseq import trimseq


def polymul(c1, c2):
    return trimseq(numpy.convolve(*as_series([c1, c2])))
