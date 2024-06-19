import numpy

from beignet.polynomial import _as_series
from beignet.polynomial._legcompanion import legcompanion


def legroots(c):
    [c] = _as_series([c])
    if len(c) < 2:
        return numpy.array([], dtype=c.dtype)
    if len(c) == 2:
        return numpy.array([-c[0] / c[1]])

    m = legcompanion(c)[::-1, ::-1]
    r = numpy.linalg.eigvals(m)
    r.sort()
    return r
