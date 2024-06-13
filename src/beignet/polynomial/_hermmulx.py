import numpy

from ._as_series import as_series


def hermmulx(c):
    [c] = as_series([c])

    if len(c) == 1 and c[0] == 0:
        return c

    prd = numpy.empty(len(c) + 1, dtype=c.dtype)
    prd[0] = c[0] * 0
    prd[1] = c[0] / 2
    for i in range(1, len(c)):
        prd[i + 1] = c[i] / 2
        prd[i - 1] += c[i] * i
    return prd
