import functools
import operator
import warnings

import numpy

__all__ = [
    "as_series",
    "getdomain",
    "mapdomain",
    "mapparms",
    "trimcoef",
    "trimseq",
]


class RankWarning(RuntimeWarning):
    pass


def trimseq(seq):
    if len(seq) == 0 or seq[-1] != 0:
        return seq
    else:
        for i in range(len(seq) - 1, -1, -1):
            if seq[i] != 0:
                break
        return seq[: i + 1]


def as_series(alist, trim=True):
    arrays = [numpy.array(a, ndmin=1) for a in alist]
    for a in arrays:
        if a.size == 0:
            raise ValueError("Coefficient array is empty")
    if any(a.ndim != 1 for a in arrays):
        raise ValueError("Coefficient array is not 1-d")
    if trim:
        arrays = [trimseq(a) for a in arrays]

    if any(a.dtype == numpy.dtype(object) for a in arrays):
        ret = []
        for a in arrays:
            if a.dtype != numpy.dtype(object):
                tmp = numpy.empty(len(a), dtype=numpy.dtype(object))
                tmp[:] = a[:]
                ret.append(tmp)
            else:
                ret.append(a.copy())
    else:
        try:
            dtype = numpy.common_type(*arrays)
        except Exception as e:
            raise ValueError("Coefficient arrays have no common type") from e
        ret = [numpy.array(a, copy=True, dtype=dtype) for a in arrays]
    return ret


def trimcoef(c, tol=0):
    if tol < 0:
        raise ValueError("tol must be non-negative")

    [c] = as_series([c])
    [ind] = numpy.nonzero(numpy.abs(c) > tol)
    if len(ind) == 0:
        return c[:1] * 0
    else:
        return c[: ind[-1] + 1].copy()


def getdomain(x):
    [x] = as_series([x], trim=False)
    if x.dtype.char in numpy.typecodes["Complex"]:
        rmin, rmax = x.real.min(), x.real.max()
        imin, imax = x.imag.min(), x.imag.max()
        return numpy.array((complex(rmin, imin), complex(rmax, imax)))
    else:
        return numpy.array((x.min(), x.max()))


def mapparms(old, new):
    oldlen = old[1] - old[0]
    newlen = new[1] - new[0]
    off = (old[1] * new[0] - old[0] * new[1]) / oldlen
    scl = newlen / oldlen
    return off, scl


def mapdomain(x, old, new):
    x = numpy.asanyarray(x)
    off, scl = mapparms(old, new)
    return off + scl * x


def _nth_slice(i, ndim):
    sl = [numpy.newaxis] * ndim
    sl[i] = slice(None)
    return tuple(sl)


def _vander_nd(vander_fs, points, degrees):
    n_dims = len(vander_fs)
    if n_dims != len(points):
        raise ValueError(
            f"Expected {n_dims} dimensions of sample points, got {len(points)}"
        )
    if n_dims != len(degrees):
        raise ValueError(f"Expected {n_dims} dimensions of degrees, got {len(degrees)}")
    if n_dims == 0:
        raise ValueError("Unable to guess a dtype or shape when no points are given")

    points = tuple(numpy.asarray(tuple(points)) + 0.0)

    vander_arrays = (
        vander_fs[i](points[i], degrees[i])[(...,) + _nth_slice(i, n_dims)]
        for i in range(n_dims)
    )

    return functools.reduce(operator.mul, vander_arrays)


def _vander_nd_flat(vander_fs, points, degrees):
    v = _vander_nd(vander_fs, points, degrees)
    return v.reshape(v.shape[: -len(degrees)] + (-1,))


def _fromroots(line_f, mul_f, roots):
    if len(roots) == 0:
        return numpy.ones(1)
    else:
        [roots] = as_series([roots], trim=False)
        roots.sort()
        p = [line_f(-r, 1) for r in roots]
        n = len(p)
        while n > 1:
            m, r = divmod(n, 2)
            tmp = [mul_f(p[i], p[i + m]) for i in range(m)]
            if r:
                tmp[0] = mul_f(tmp[0], p[-1])
            p = tmp
            n = m
        return p[0]


def _valnd(val_f, c, *args):
    args = [numpy.asanyarray(a) for a in args]
    shape0 = args[0].shape
    if not all((a.shape == shape0 for a in args[1:])):
        if len(args) == 3:
            raise ValueError("x, y, z are incompatible")
        elif len(args) == 2:
            raise ValueError("x, y are incompatible")
        else:
            raise ValueError("ordinates are incompatible")
    it = iter(args)
    x0 = next(it)

    c = val_f(x0, c)
    for xi in it:
        c = val_f(xi, c, tensor=False)
    return c


def _gridnd(val_f, c, *args):
    for xi in args:
        c = val_f(xi, c)
    return c


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


def _add(c1, c2):
    [c1, c2] = as_series([c1, c2])
    if len(c1) > len(c2):
        c1[: c2.size] += c2
        ret = c1
    else:
        c2[: c1.size] += c1
        ret = c2
    return trimseq(ret)


def _sub(c1, c2):
    [c1, c2] = as_series([c1, c2])
    if len(c1) > len(c2):
        c1[: c2.size] -= c2
        ret = c1
    else:
        c2 = -c2
        c2[: c1.size] += c1
        ret = c2
    return trimseq(ret)


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


def _pow(mul_f, c, pow, maxpower):
    [c] = as_series([c])
    power = int(pow)
    if power != pow or power < 0:
        raise ValueError("Power must be a non-negative integer.")
    elif maxpower is not None and power > maxpower:
        raise ValueError("Power is too large")
    elif power == 0:
        return numpy.array([1], dtype=c.dtype)
    elif power == 1:
        return c
    else:
        prd = c
        for _ in range(2, power + 1):
            prd = mul_f(prd, c)
        return prd


def _as_int(x, desc):
    try:
        return operator.index(x)
    except TypeError as e:
        raise TypeError(f"{desc} must be an integer, received {x}") from e
