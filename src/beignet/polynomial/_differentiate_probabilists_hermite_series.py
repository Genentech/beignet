import operator

import numpy
import torch

from .__normalize_axis_index import _normalize_axis_index


def differentiate_probabilists_hermite_series(c, m=1, scl=1, axis=0):
    c = torch.ravel(c)
    if c.dtype.char in "?bBhHiIlLqQpP":
        c = c.astype(numpy.double)
    cnt = operator.index(m)
    iaxis = operator.index(axis)
    if cnt < 0:
        raise ValueError("The order of derivation must be non-negative")
    iaxis = _normalize_axis_index(iaxis, c.ndim)

    if cnt == 0:
        return c

    c = numpy.moveaxis(c, iaxis, 0)
    n = len(c)
    if cnt >= n:
        return c[:1] * 0
    else:
        for _ in range(cnt):
            n = n - 1
            c *= scl
            der = numpy.empty((n,) + c.shape[1:], dtype=c.dtype)
            for j in range(n, 0, -1):
                der[j - 1] = j * c[j]
            c = der
    c = numpy.moveaxis(c, 0, iaxis)
    return c
