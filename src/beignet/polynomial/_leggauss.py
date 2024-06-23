import operator

import numpy

from beignet.polynomial._differentiate_legendre_series import (
    differentiate_legendre_series,
)
from beignet.polynomial._legendre_series_companion import legendre_series_companion

from ._evaluate_legendre_series_1d import evaluate_legendre_series_1d


def leggauss(input):
    ideg = operator.index(input)
    if ideg <= 0:
        raise ValueError("deg must be a positive integer")

    c = numpy.array([0] * input + [1])
    m = legendre_series_companion(c)
    x = numpy.linalg.eigvalsh(m)

    dy = evaluate_legendre_series_1d(x, c)
    df = evaluate_legendre_series_1d(x, differentiate_legendre_series(c))
    x -= dy / df

    fm = evaluate_legendre_series_1d(x, c[1:])
    fm /= numpy.abs(fm).max()
    df /= numpy.abs(df).max()
    weights = 1 / (fm * df)

    weights = (weights + weights[::-1]) / 2
    x = (x - x[::-1]) / 2

    weights = weights * (2.0 / weights.sum())

    return x, weights
