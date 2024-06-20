import operator

import numpy

from beignet.polynomial._legcompanion import legcompanion
from beignet.polynomial._legder import legder

from ._evaluate_1d_legendre_series import evaluate_1d_legendre_series


def leggauss(input):
    ideg = operator.index(input)
    if ideg <= 0:
        raise ValueError("deg must be a positive integer")

    c = numpy.array([0] * input + [1])
    m = legcompanion(c)
    x = numpy.linalg.eigvalsh(m)

    dy = evaluate_1d_legendre_series(x, c)
    df = evaluate_1d_legendre_series(x, legder(c))
    x -= dy / df

    fm = evaluate_1d_legendre_series(x, c[1:])
    fm /= numpy.abs(fm).max()
    df /= numpy.abs(df).max()
    weights = 1 / (fm * df)

    weights = (weights + weights[::-1]) / 2
    x = (x - x[::-1]) / 2

    weights = weights * (2.0 / weights.sum())

    return x, weights
