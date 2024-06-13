import numpy

from ._as_series import as_series


def polycompanion(c):
    [c] = as_series([c])

    if len(c) < 2:
        raise ValueError("Series must have maximum degree of at least 1.")

    if len(c) == 2:
        return numpy.array([[-c[0] / c[1]]])

    n = len(c) - 1

    mat = numpy.zeros((n, n), dtype=c.dtype)

    bot = mat.reshape(-1)[n :: n + 1]

    bot[...] = 1

    mat[:, -1] -= c[:-1] / c[-1]

    return mat
