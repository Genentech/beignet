import numpy

from .__deprecate_as_int import _deprecate_as_int
from ._chebval import chebval
from ._normalize_axis_index import normalize_axis_index


def chebint(c, m=1, k=None, lbnd=0, scl=1, axis=0):
    if k is None:
        k = []
    c = numpy.array(c, ndmin=1, copy=True)
    if c.dtype.char in "?bBhHiIlLqQpP":
        c = c.astype(numpy.double)
    if not numpy.iterable(k):
        k = [k]
    cnt = _deprecate_as_int(m, "the order of integration")
    iaxis = _deprecate_as_int(axis, "the axis")
    if cnt < 0:
        raise ValueError("The order of integration must be non-negative")
    if len(k) > cnt:
        raise ValueError("Too many integration constants")
    if numpy.ndim(lbnd) != 0:
        raise ValueError("lbnd must be a scalar.")
    if numpy.ndim(scl) != 0:
        raise ValueError("scl must be a scalar.")
    iaxis = normalize_axis_index(iaxis, c.ndim)

    if cnt == 0:
        return c

    c = numpy.moveaxis(c, iaxis, 0)
    k = list(k) + [0] * (cnt - len(k))
    for i in range(cnt):
        n = len(c)
        c *= scl
        if n == 1 and numpy.all(c[0] == 0):
            c[0] += k[i]
        else:
            tmp = numpy.empty((n + 1,) + c.shape[1:], dtype=c.dtype)
            tmp[0] = c[0] * 0
            tmp[1] = c[0]
            if n > 1:
                tmp[2] = c[1] / 4
            for j in range(2, n):
                tmp[j + 1] = c[j] / (2 * (j + 1))
                tmp[j - 1] -= c[j] / (2 * (j - 1))
            tmp[0] += k[i] - chebval(lbnd, tmp)
            c = tmp
    c = numpy.moveaxis(c, 0, iaxis)
    return c
