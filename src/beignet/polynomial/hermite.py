import numpy
import numpy.linalg

from ._normalize_axis_index import normalize_axis_index
from ._polybase import ABCPolyBase
from .polyutils import (
    _add,
    _deprecate_as_int,
    _div,
    _fit,
    _fromroots,
    _gridnd,
    _pow,
    _sub,
    _valnd,
    _vander_nd_flat,
    as_series,
    trimcoef,
)

hermtrim = trimcoef


def poly2herm(pol):
    [pol] = as_series([pol])
    deg = len(pol) - 1
    res = 0
    for i in range(deg, -1, -1):
        res = hermadd(hermmulx(res), pol[i])
    return res


def herm2poly(c):
    from .polynomial import polyadd, polymulx, polysub

    [c] = as_series([c])
    n = len(c)
    if n == 1:
        return c
    if n == 2:
        c[1] *= 2
        return c
    else:
        c0 = c[-2]
        c1 = c[-1]

        for i in range(n - 1, 1, -1):
            tmp = c0
            c0 = polysub(c[i - 2], c1 * (2 * (i - 1)))
            c1 = polyadd(tmp, polymulx(c1) * 2)
        return polyadd(c0, polymulx(c1) * 2)


hermdomain = numpy.array([-1, 1])


hermzero = numpy.array([0])


hermone = numpy.array([1])


hermx = numpy.array([0, 1 / 2])


def hermline(off, scl):
    if scl != 0:
        return numpy.array([off, scl / 2])
    else:
        return numpy.array([off])


def hermfromroots(roots):
    return _fromroots(hermline, hermmul, roots)


def hermadd(c1, c2):
    return _add(c1, c2)


def hermsub(c1, c2):
    return _sub(c1, c2)


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


def hermmul(c1, c2):
    [c1, c2] = as_series([c1, c2])

    if len(c1) > len(c2):
        c = c2
        xs = c1
    else:
        c = c1
        xs = c2

    if len(c) == 1:
        c0 = c[0] * xs
        c1 = 0
    elif len(c) == 2:
        c0 = c[0] * xs
        c1 = c[1] * xs
    else:
        nd = len(c)
        c0 = c[-2] * xs
        c1 = c[-1] * xs
        for i in range(3, len(c) + 1):
            tmp = c0
            nd = nd - 1
            c0 = hermsub(c[-i] * xs, c1 * (2 * (nd - 1)))
            c1 = hermadd(tmp, hermmulx(c1) * 2)
    return hermadd(c0, hermmulx(c1) * 2)


def hermdiv(c1, c2):
    return _div(hermmul, c1, c2)


def hermpow(c, pow, maxpower=16):
    return _pow(hermmul, c, pow, maxpower)


def hermder(c, m=1, scl=1, axis=0):
    c = numpy.array(c, ndmin=1, copy=True)
    if c.dtype.char in "?bBhHiIlLqQpP":
        c = c.astype(numpy.double)
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
            der = numpy.empty((n,) + c.shape[1:], dtype=c.dtype)
            for j in range(n, 0, -1):
                der[j - 1] = (2 * j) * c[j]
            c = der
    c = numpy.moveaxis(c, 0, iaxis)
    return c


def hermint(c, m=1, k=None, lbnd=0, scl=1, axis=0):
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
            tmp[1] = c[0] / 2
            for j in range(1, n):
                tmp[j + 1] = c[j] / (2 * (j + 1))
            tmp[0] += k[i] - hermval(lbnd, tmp)
            c = tmp
    c = numpy.moveaxis(c, 0, iaxis)
    return c


def hermval(x, c, tensor=True):
    c = numpy.array(c, ndmin=1, copy=False)
    if c.dtype.char in "?bBhHiIlLqQpP":
        c = c.astype(numpy.double)
    if isinstance(x, (tuple, list)):
        x = numpy.asarray(x)
    if isinstance(x, numpy.ndarray) and tensor:
        c = c.reshape(c.shape + (1,) * x.ndim)

    x2 = x * 2
    if len(c) == 1:
        c0 = c[0]
        c1 = 0
    elif len(c) == 2:
        c0 = c[0]
        c1 = c[1]
    else:
        nd = len(c)
        c0 = c[-2]
        c1 = c[-1]
        for i in range(3, len(c) + 1):
            tmp = c0
            nd = nd - 1
            c0 = c[-i] - c1 * (2 * (nd - 1))
            c1 = tmp + c1 * x2
    return c0 + c1 * x2


def hermval2d(x, y, c):
    return _valnd(hermval, c, x, y)


def hermgrid2d(x, y, c):
    return _gridnd(hermval, c, x, y)


def hermval3d(x, y, z, c):
    return _valnd(hermval, c, x, y, z)


def hermgrid3d(x, y, z, c):
    return _gridnd(hermval, c, x, y, z)


def hermvander(x, deg):
    ideg = _deprecate_as_int(deg, "deg")
    if ideg < 0:
        raise ValueError("deg must be non-negative")

    x = numpy.array(x, copy=False, ndmin=1) + 0.0
    dims = (ideg + 1,) + x.shape
    dtyp = x.dtype
    v = numpy.empty(dims, dtype=dtyp)
    v[0] = x * 0 + 1
    if ideg > 0:
        x2 = x * 2
        v[1] = x2
        for i in range(2, ideg + 1):
            v[i] = v[i - 1] * x2 - v[i - 2] * (2 * (i - 1))
    return numpy.moveaxis(v, 0, -1)


def hermvander2d(x, y, deg):
    return _vander_nd_flat((hermvander, hermvander), (x, y), deg)


def hermvander3d(x, y, z, deg):
    return _vander_nd_flat((hermvander, hermvander, hermvander), (x, y, z), deg)


def hermfit(x, y, deg, rcond=None, full=False, w=None):
    return _fit(hermvander, x, y, deg, rcond, full, w)


def hermcompanion(c):
    [c] = as_series([c])
    if len(c) < 2:
        raise ValueError("Series must have maximum degree of at least 1.")
    if len(c) == 2:
        return numpy.array([[-0.5 * c[0] / c[1]]])

    n = len(c) - 1
    mat = numpy.zeros((n, n), dtype=c.dtype)
    scl = numpy.hstack((1.0, 1.0 / numpy.sqrt(2.0 * numpy.arange(n - 1, 0, -1))))
    scl = numpy.multiply.accumulate(scl)[::-1]
    top = mat.reshape(-1)[1 :: n + 1]
    bot = mat.reshape(-1)[n :: n + 1]
    top[...] = numpy.sqrt(0.5 * numpy.arange(1, n))
    bot[...] = top
    mat[:, -1] -= scl * c[:-1] / (2.0 * c[-1])
    return mat


def hermroots(c):
    [c] = as_series([c])
    if len(c) <= 1:
        return numpy.array([], dtype=c.dtype)
    if len(c) == 2:
        return numpy.array([-0.5 * c[0] / c[1]])

    m = hermcompanion(c)[::-1, ::-1]
    r = numpy.linalg.eigvals(m)
    r.sort()
    return r


def _normed_hermite_n(x, n):
    if n == 0:
        return numpy.full(x.shape, 1 / numpy.sqrt(numpy.sqrt(numpy.pi)))

    c0 = 0.0
    c1 = 1.0 / numpy.sqrt(numpy.sqrt(numpy.pi))
    nd = float(n)
    for _ in range(n - 1):
        tmp = c0
        c0 = -c1 * numpy.sqrt((nd - 1.0) / nd)
        c1 = tmp + c1 * x * numpy.sqrt(2.0 / nd)
        nd = nd - 1.0
    return c0 + c1 * x * numpy.sqrt(2)


def hermgauss(deg):
    ideg = _deprecate_as_int(deg, "deg")
    if ideg <= 0:
        raise ValueError("deg must be a positive integer")

    c = numpy.array([0] * deg + [1], dtype=numpy.float64)
    m = hermcompanion(c)
    x = numpy.linalg.eigvalsh(m)

    dy = _normed_hermite_n(x, ideg)
    df = _normed_hermite_n(x, ideg - 1) * numpy.sqrt(2 * ideg)
    x -= dy / df

    fm = _normed_hermite_n(x, ideg - 1)
    fm /= numpy.abs(fm).max()
    w = 1 / (fm * fm)

    w = (w + w[::-1]) / 2
    x = (x - x[::-1]) / 2

    w *= numpy.sqrt(numpy.pi) / w.sum()

    return x, w


def hermweight(x):
    w = numpy.exp(-(x**2))
    return w


class Hermite(ABCPolyBase):
    _add = staticmethod(hermadd)
    _sub = staticmethod(hermsub)
    _mul = staticmethod(hermmul)
    _div = staticmethod(hermdiv)
    _pow = staticmethod(hermpow)
    _val = staticmethod(hermval)
    _int = staticmethod(hermint)
    _der = staticmethod(hermder)
    _fit = staticmethod(hermfit)
    _line = staticmethod(hermline)
    _roots = staticmethod(hermroots)
    _fromroots = staticmethod(hermfromroots)

    domain = numpy.array(hermdomain)
    window = numpy.array(hermdomain)
    basis_name = "H"
