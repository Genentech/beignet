import numpy

from ._as_series import as_series


def trimcoef(c, tol=0):
    if tol < 0:
        raise ValueError("tol must be non-negative")

    [c] = as_series([c])
    [ind] = numpy.nonzero(numpy.abs(c) > tol)
    if len(ind) == 0:
        return c[:1] * 0
    else:
        return c[: ind[-1] + 1].copy()
