import operator

import numpy
import torch

from .__normalize_axis_index import _normalize_axis_index


def differentiate_chebyshev_series(c, m=1, scl=1, axis=0):
    c = torch.ravel(c)

    if c.dtype.char in "?bBhHiIlLqQpP":
        c = c.astype(numpy.double)

    cnt = operator.index(m)

    iaxis = operator.index(axis)

    if cnt < 0:
        raise ValueError

    iaxis = _normalize_axis_index(iaxis, c.ndim)

    if cnt == 0:
        return c

    c = numpy.moveaxis(c, iaxis, 0)

    n = len(c)

    if cnt >= n:
        c = c[:1] * 0
    else:
        for _ in range(cnt):
            n = n - 1

            c *= scl

            der = numpy.empty((n,) + c.shape[1:], dtype=c.dtype)

            for j in range(n, 2, -1):
                der[j - 1] = (2 * j) * c[j]

                c[j - 2] += (j * c[j]) / (j - 2)

            if n > 1:
                der[1] = 4 * c[2]

            der[0] = c[1]

            c = der

    c = numpy.moveaxis(c, 0, iaxis)

    return c
