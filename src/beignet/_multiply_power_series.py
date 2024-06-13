import numpy

from .polynomial._as_series import as_series
from .polynomial._trimseq import trimseq


def multiply_power_series(c1, c2):
    return trimseq(numpy.convolve(*as_series([c1, c2])))
