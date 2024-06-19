import numpy
import torch

from beignet.polynomial import _as_series
from beignet.polynomial._chebcompanion import chebcompanion


def chebroots(c):
    [c] = _as_series([c])
    if len(c) < 2:
        return torch.tensor([], dtype=c.dtype)
    if len(c) == 2:
        return torch.tensor([-c[0] / c[1]])

    m = chebcompanion(c)[::-1, ::-1]
    r = numpy.linalg.eigvals(m)
    r.sort()
    return r
