import numpy
import torch

from .__as_series import _as_series


def physicists_hermite_series_companion(input):
    (input,) = _as_series([input])

    if len(input) < 2:
        raise ValueError

    if len(input) == 2:
        return torch.tensor([[-0.5 * input[0] / input[1]]])

    n = len(input) - 1

    mat = numpy.zeros((n, n), dtype=input.dtype)

    scl = numpy.hstack((1.0, 1.0 / numpy.sqrt(2.0 * numpy.arange(n - 1, 0, -1))))

    scl = numpy.multiply.accumulate(scl)[::-1]

    top = mat.reshape(-1)[1 :: n + 1]

    bot = mat.reshape(-1)[n :: n + 1]

    top[...] = numpy.sqrt(0.5 * numpy.arange(1, n))

    bot[...] = top

    mat[:, -1] = mat[:, -1] - (scl * input[:-1] / (2.0 * input[-1]))

    return mat
