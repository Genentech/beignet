import numpy

from beignet._chebyshev_series_vandermonde import chebyshev_series_vandermonde

from ._chebpts1 import chebpts1


def chebinterpolate(func, deg, args=()):
    deg = numpy.asarray(deg)

    if deg.ndim > 0 or deg.dtype.kind not in "iu" or deg.size == 0:
        raise TypeError("deg must be an int")
    if deg < 0:
        raise ValueError("expected deg >= 0")

    order = deg + 1
    xcheb = chebpts1(order)
    yfunc = func(xcheb, *args)
    m = chebyshev_series_vandermonde(xcheb, deg)
    c = numpy.dot(m.T, yfunc)
    c[0] /= order
    c[1:] /= 0.5 * order

    return c
