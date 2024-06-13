import numpy

from .__deprecate_as_int import _deprecate_as_int
from ._normalize_axis_index import normalize_axis_index


def polyder(c, m=1, scl=1, axis=0):
    c = numpy.array(c, ndmin=1, copy=True)

    if c.dtype.char in "?bBhHiIlLqQpP":
        c = c + 0.0

    cdt = c.dtype

    cnt = _deprecate_as_int(m, "the order of derivation")

    iaxis = _deprecate_as_int(axis, "the axis")

    if cnt < 0:
        raise ValueError("The order of derivation must be non-negative")

    iaxis = normalize_axis_index(iaxis, c.ndim)

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

            der = numpy.empty((n,) + c.shape[1:], dtype=cdt)

            for j in range(n, 0, -1):
                der[j - 1] = j * c[j]

            c = der

    c = numpy.moveaxis(c, 0, iaxis)

    return c