import warnings

import numpy


class RankWarning(RuntimeWarning):
    pass


def _fit(func, x, y, degree, relative_condition=None, full=False, weight=None):
    x = numpy.asarray(x) + 0.0
    y = numpy.asarray(y) + 0.0

    degree = numpy.asarray(degree)

    if degree.ndim > 1 or degree.dtype.kind not in "iu" or degree.size == 0:
        raise TypeError("deg must be an int or non-empty 1-D array of int")
    if degree.min() < 0:
        raise ValueError("expected deg >= 0")
    if x.ndim != 1:
        raise TypeError("expected 1D vector for x")
    if x.size == 0:
        raise TypeError("expected non-empty vector for x")
    if y.ndim < 1 or y.ndim > 2:
        raise TypeError("expected 1D or 2D array for y")
    if len(x) != len(y):
        raise TypeError("expected x and y to have same length")

    if degree.ndim == 0:
        lmax = degree
        order = lmax + 1
        van = func(x, lmax)
    else:
        degree = numpy.sort(degree)
        lmax = degree[-1]
        order = len(degree)
        van = func(x, lmax)[:, degree]

    lhs = van.T
    rhs = y.T
    if weight is not None:
        weight = numpy.asarray(weight) + 0.0
        if weight.ndim != 1:
            raise TypeError("expected 1D vector for w")
        if len(x) != len(weight):
            raise TypeError("expected x and w to have same length")

        lhs = lhs * weight
        rhs = rhs * weight

    if relative_condition is None:
        relative_condition = len(x) * numpy.finfo(x.dtype).eps

    if issubclass(lhs.dtype.type, numpy.complexfloating):
        scl = numpy.sqrt((numpy.square(lhs.real) + numpy.square(lhs.imag)).sum(1))
    else:
        scl = numpy.sqrt(numpy.square(lhs).sum(1))

    scl[scl == 0] = 1

    c, resids, rank, s = numpy.linalg.lstsq(lhs.T / scl, rhs.T, relative_condition)
    c = (c.T / scl).T

    if degree.ndim > 0:
        if c.ndim == 2:
            cc = numpy.zeros((lmax + 1, c.shape[1]), dtype=c.dtype)
        else:
            cc = numpy.zeros(lmax + 1, dtype=c.dtype)

        cc[degree] = c

        c = cc

    if rank != order and not full:
        msg = "The fit may be poorly conditioned"

        warnings.warn(msg, RankWarning, stacklevel=2)

    if full:
        return c, [resids, rank, s, relative_condition]
    else:
        return c
