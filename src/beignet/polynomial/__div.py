import numpy

from ._as_series import as_series
from ._trimseq import trimseq


def _div(mul_f, c1, c2):
    [c1, c2] = as_series([c1, c2])
    if c2[-1] == 0:
        raise ZeroDivisionError()

    lc1 = len(c1)
    lc2 = len(c2)
    if lc1 < lc2:
        return c1[:1] * 0, c1
    elif lc2 == 1:
        return c1 / c2[-1], c1[:1] * 0
    else:
        quo = numpy.empty(lc1 - lc2 + 1, dtype=c1.dtype)
        rem = c1
        for i in range(lc1 - lc2, -1, -1):
            p = mul_f([0] * i + [1], c2)
            q = rem[-1] / p[-1]
            rem = rem[:-1] - q * p[:-1]
            quo[i] = q
        return quo, trimseq(rem)
