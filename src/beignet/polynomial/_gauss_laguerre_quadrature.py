import operator

import numpy
from torch import Tensor

from beignet.polynomial._differentiate_laguerre_series import (
    differentiate_laguerre_series,
)
from beignet.polynomial._laguerre_series_companion import laguerre_series_companion

from ._evaluate_laguerre_series_1d import evaluate_laguerre_series_1d


def gauss_laguerre_quadrature(input: Tensor) -> (Tensor, Tensor):
    ideg = operator.index(input)
    if ideg <= 0:
        raise ValueError

    c = numpy.array([0] * input + [1])
    m = laguerre_series_companion(c)
    x = numpy.linalg.eigvalsh(m)

    dy = evaluate_laguerre_series_1d(x, c)
    df = evaluate_laguerre_series_1d(x, differentiate_laguerre_series(c))
    x -= dy / df

    fm = evaluate_laguerre_series_1d(x, c[1:])

    fm = fm / numpy.max(numpy.abs(fm))
    df = df / numpy.max(numpy.abs(df))

    weight = 1.0 / (fm * df)

    weight = weight / numpy.sum(weight)

    return x, weight
