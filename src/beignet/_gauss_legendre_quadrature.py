import numpy

from beignet._differentiate_legendre_series import differentiate_legendre_series
from beignet._evaluate_legendre_series import evaluate_legendre_series

from .polynomial.__deprecate_as_int import _deprecate_as_int
from .polynomial._legcompanion import legcompanion


def gauss_legendre_quadrature(deg):
    ideg = _deprecate_as_int(deg, "deg")
    if ideg <= 0:
        raise ValueError("deg must be a positive integer")

    c = numpy.array([0] * deg + [1])
    m = legcompanion(c)
    x = numpy.linalg.eigvalsh(m)

    dy = evaluate_legendre_series(x, c)
    df = evaluate_legendre_series(x, differentiate_legendre_series(c))
    x -= dy / df

    fm = evaluate_legendre_series(x, c[1:])
    fm /= numpy.abs(fm).max()
    df /= numpy.abs(df).max()
    w = 1 / (fm * df)

    w = (w + w[::-1]) / 2
    x = (x - x[::-1]) / 2

    w *= 2.0 / w.sum()

    return x, w
