import operator

import numpy

from beignet.polynomial._differentiate_laguerre_series import (
    differentiate_laguerre_series,
)
from beignet.polynomial._laguerre_series_companion import laguerre_series_companion

from ._evaluate_1d_laguerre_series import evaluate_1d_laguerre_series


def laggauss(input):
    ideg = operator.index(input)
    if ideg <= 0:
        raise ValueError("deg must be a positive integer")

    c = numpy.array([0] * input + [1])
    m = laguerre_series_companion(c)
    x = numpy.linalg.eigvalsh(m)

    dy = evaluate_1d_laguerre_series(x, c)
    df = evaluate_1d_laguerre_series(x, differentiate_laguerre_series(c))
    x -= dy / df

    fm = evaluate_1d_laguerre_series(x, c[1:])

    fm = fm / numpy.max(numpy.abs(fm))
    df = df / numpy.max(numpy.abs(df))

    weight = 1.0 / (fm * df)

    weight = weight / numpy.sum(weight)

    return x, weight
