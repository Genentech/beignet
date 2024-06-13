import numpy

from beignet._differentiate_laguerre_series import differentiate_laguerre_series
from beignet._evaluate_laguerre_series import evaluate_laguerre_series

from .polynomial.__deprecate_as_int import _deprecate_as_int
from .polynomial._lagcompanion import lagcompanion


def gauss_laguerre_quadrature(deg):
    ideg = _deprecate_as_int(deg, "deg")
    if ideg <= 0:
        raise ValueError("deg must be a positive integer")

    c = numpy.array([0] * deg + [1])
    m = lagcompanion(c)
    x = numpy.linalg.eigvalsh(m)

    dy = evaluate_laguerre_series(x, c)
    df = evaluate_laguerre_series(x, differentiate_laguerre_series(c))
    x -= dy / df

    fm = evaluate_laguerre_series(x, c[1:])
    fm /= numpy.abs(fm).max()
    df /= numpy.abs(df).max()
    w = 1 / (fm * df)

    w /= w.sum()

    return x, w
