import numpy

from ._as_series import as_series
from ._hermcompanion import hermcompanion


def hermroots(c):
    [c] = as_series([c])
    if len(c) <= 1:
        return numpy.array([], dtype=c.dtype)
    if len(c) == 2:
        return numpy.array([-0.5 * c[0] / c[1]])

    m = hermcompanion(c)[::-1, ::-1]
    r = numpy.linalg.eigvals(m)
    r.sort()
    return r
