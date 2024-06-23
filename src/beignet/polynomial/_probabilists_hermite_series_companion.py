import numpy
import torch

from .__as_series import _as_series


def probabilists_hermite_series_companion(c):
    [c] = _as_series([c])
    if len(c) < 2:
        raise ValueError("Series must have maximum degree of at least 1.")
    if len(c) == 2:
        return torch.tensor([[-c[0] / c[1]]])

    n = len(c) - 1
    mat = numpy.zeros((n, n), dtype=c.dtype)
    scl = numpy.hstack((1.0, 1.0 / numpy.sqrt(numpy.arange(n - 1, 0, -1))))
    scl = numpy.multiply.accumulate(scl)[::-1]
    top = mat.reshape(-1)[1 :: n + 1]
    bot = mat.reshape(-1)[n :: n + 1]
    top[...] = numpy.sqrt(numpy.arange(1, n))
    bot[...] = top
    mat[:, -1] -= scl * c[:-1] / c[-1]
    return mat
