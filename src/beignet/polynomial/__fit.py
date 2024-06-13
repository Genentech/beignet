import warnings

import numpy

from ._rank_warning import RankWarning


def _fit(vander_f, x, y, deg, rcond=None, full=False, w=None):
    x = numpy.asarray(x) + 0.0
    y = numpy.asarray(y) + 0.0
    deg = numpy.asarray(deg)

    if deg.ndim > 1 or deg.dtype.kind not in "iu" or deg.size == 0:
        raise TypeError("deg must be an int or non-empty 1-D array of int")
    if deg.min() < 0:
        raise ValueError("expected deg >= 0")
    if x.ndim != 1:
        raise TypeError("expected 1D vector for x")
    if x.size == 0:
        raise TypeError("expected non-empty vector for x")
    if y.ndim < 1 or y.ndim > 2:
        raise TypeError("expected 1D or 2D array for y")
    if len(x) != len(y):
        raise TypeError("expected x and y to have same length")

    if deg.ndim == 0:
        lmax = deg
        order = lmax + 1
        van = vander_f(x, lmax)
    else:
        deg = numpy.sort(deg)
        lmax = deg[-1]
        order = len(deg)
        van = vander_f(x, lmax)[:, deg]

    lhs = van.T
    rhs = y.T
    if w is not None:
        w = numpy.asarray(w) + 0.0
        if w.ndim != 1:
            raise TypeError("expected 1D vector for w")
        if len(x) != len(w):
            raise TypeError("expected x and w to have same length")

        lhs = lhs * w
        rhs = rhs * w

    if rcond is None:
        rcond = len(x) * numpy.finfo(x.dtype).eps

    if issubclass(lhs.dtype.type, numpy.complexfloating):
        scl = numpy.sqrt((numpy.square(lhs.real) + numpy.square(lhs.imag)).sum(1))
    else:
        scl = numpy.sqrt(numpy.square(lhs).sum(1))
    scl[scl == 0] = 1

    c, resids, rank, s = numpy.linalg.lstsq(lhs.T / scl, rhs.T, rcond)
    c = (c.T / scl).T

    if deg.ndim > 0:
        if c.ndim == 2:
            cc = numpy.zeros((lmax + 1, c.shape[1]), dtype=c.dtype)
        else:
            cc = numpy.zeros(lmax + 1, dtype=c.dtype)
        cc[deg] = c
        c = cc

    if rank != order and not full:
        msg = "The fit may be poorly conditioned"
        warnings.warn(msg, RankWarning, stacklevel=2)

    if full:
        return c, [resids, rank, s, rcond]
    else:
        return c
