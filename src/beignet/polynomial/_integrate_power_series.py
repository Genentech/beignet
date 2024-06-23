import operator

import numpy

from .__normalize_axis_index import _normalize_axis_index
from ._evaluate_1d_power_series import evaluate_1d_power_series


def integrate_power_series(c, m=1, k=None, lbnd=0, scl=1, axis=0):
    if k is None:
        k = []
    c = numpy.array(c, ndmin=1)
    if c.dtype.char in "?bBhHiIlLqQpP":
        c = c + 0.0
    cdt = c.dtype
    if not numpy.iterable(k):
        k = [k]
    cnt = operator.index(m)
    iaxis = operator.index(axis)
    if cnt < 0:
        raise ValueError("The order of integration must be non-negative")
    if len(k) > cnt:
        raise ValueError("Too many integration constants")
    if numpy.ndim(lbnd) != 0:
        raise ValueError("lbnd must be a scalar.")
    if numpy.ndim(scl) != 0:
        raise ValueError("scl must be a scalar.")
    iaxis = _normalize_axis_index(iaxis, c.ndim)

    if cnt == 0:
        return c

    k = list(k) + [0] * (cnt - len(k))
    c = numpy.moveaxis(c, iaxis, 0)
    for i in range(cnt):
        n = len(c)
        c *= scl
        if n == 1 and numpy.all(c[0] == 0):
            c[0] += k[i]
        else:
            tmp = numpy.empty((n + 1,) + c.shape[1:], dtype=cdt)
            tmp[0] = c[0] * 0
            tmp[1] = c[0]
            for j in range(1, n):
                tmp[j + 1] = c[j] / (j + 1)
            tmp[0] += k[i] - evaluate_1d_power_series(lbnd, tmp)
            c = tmp
    c = numpy.moveaxis(c, 0, iaxis)
    return c
