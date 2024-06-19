import numpy

from beignet.polynomial._lagcompanion import lagcompanion

from .__as_series import _as_series


def lagroots(c):
    [c] = _as_series([c])
    if len(c) <= 1:
        return numpy.array([], dtype=c.dtype)
    if len(c) == 2:
        return numpy.array([1 + c[0] / c[1]])

    m = lagcompanion(c)[::-1, ::-1]
    r = numpy.linalg.eigvals(m)
    r.sort()
    return r
