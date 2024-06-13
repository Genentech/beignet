import numpy

from .polynomial._as_series import as_series
from .polynomial._legcompanion import legcompanion


def legendre_series_roots(c):
    [c] = as_series([c])
    if len(c) < 2:
        return numpy.array([], dtype=c.dtype)
    if len(c) == 2:
        return numpy.array([-c[0] / c[1]])

    m = legcompanion(c)[::-1, ::-1]
    r = numpy.linalg.eigvals(m)
    r.sort()
    return r
