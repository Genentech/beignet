import operator

import numpy
from torch import Tensor

from .__normed_hermite_e_n import _normed_hermite_e_n
from ._probabilists_hermite_series_companion import (
    probabilists_hermite_series_companion,
)


def gauss_probabilists_hermite_series_quadrature(
    input: Tensor,
) -> (Tensor, Tensor):
    ideg = operator.index(input)

    if ideg <= 0:
        raise ValueError

    c = numpy.array([0] * input + [1])
    m = probabilists_hermite_series_companion(c)
    output = numpy.linalg.eigvalsh(m)

    dy = _normed_hermite_e_n(output, ideg)
    df = _normed_hermite_e_n(output, ideg - 1) * numpy.sqrt(ideg)
    output -= dy / df

    fm = _normed_hermite_e_n(output, ideg - 1)
    fm /= numpy.abs(fm).max()

    weight = 1 / (fm * fm)

    weight = (weight + weight[::-1]) / 2

    output = (output - output[::-1]) / 2

    weight = weight * (numpy.sqrt(2 * numpy.pi) / weight.sum())

    return output, weight
