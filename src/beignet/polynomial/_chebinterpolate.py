import numpy

from ._chebyshev_nodes_1 import chebyshev_nodes_1
from ._chebyshev_series_vandermonde_1d import chebyshev_series_vandermonde_1d


def chebinterpolate(func, deg, args=()):
    deg = numpy.asarray(deg)

    if deg.ndim > 0 or deg.dtype.kind not in "iu" or deg.size == 0:
        raise TypeError("deg must be an int")
    if deg < 0:
        raise ValueError("expected deg >= 0")

    order = deg + 1
    xcheb = chebyshev_nodes_1(order)
    yfunc = func(xcheb, *args)
    m = chebyshev_series_vandermonde_1d(xcheb, deg)
    c = numpy.dot(m.T, yfunc)
    c[0] /= order
    c[1:] /= 0.5 * order

    return c
