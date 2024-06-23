import numpy

from beignet.polynomial._legendre_series_companion import legendre_series_companion

from .__as_series import _as_series


def legendre_series_roots(c):
    [c] = _as_series([c])
    if len(c) < 2:
        return numpy.array([], dtype=c.dtype)
    if len(c) == 2:
        return numpy.array([-c[0] / c[1]])

    m = legendre_series_companion(c)[::-1, ::-1]
    r = numpy.linalg.eigvals(m)
    r.sort()
    return r
