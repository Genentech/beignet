import operator

import numpy
import torch
from torch import Tensor

from .__normed_hermite_n import _normed_hermite_n
from ._physicists_hermite_series_companion import physicists_hermite_series_companion


def gauss_physicists_hermite_series_quadrature(
    input: Tensor,
) -> (Tensor, Tensor):
    ideg = operator.index(input)
    if ideg <= 0:
        raise ValueError

    c = torch.tensor([0] * input + [1], dtype=numpy.float64)
    m = physicists_hermite_series_companion(c)
    output = numpy.linalg.eigvalsh(m)

    dy = _normed_hermite_n(output, ideg)
    df = _normed_hermite_n(output, ideg - 1) * numpy.sqrt(2 * ideg)
    output -= dy / df

    fm = _normed_hermite_n(output, ideg - 1)
    fm /= numpy.abs(fm).max()
    weight = 1 / (fm * fm)

    weight = (weight + weight[::-1]) / 2
    output = (output - output[::-1]) / 2

    weight = weight * (numpy.sqrt(numpy.pi) / numpy.sum(weight))

    return output, weight
