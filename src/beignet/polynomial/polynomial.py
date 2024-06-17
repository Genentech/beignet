import numpy
import numpy.linalg
from torch._numpy._util import normalize_axis_index

from beignet.polynomial import polyutils

from ._polybase import ABCPolyBase

__all__ = [
    "Polynomial",
    "polyadd",
    "polycompanion",
    "polyder",
    "polydiv",
    "polydomain",
    "polyfit",
    "polyfromroots",
    "polygrid2d",
    "polygrid3d",
    "polyint",
    "polyline",
    "polymul",
    "polymulx",
    "polyone",
    "polypow",
    "polyroots",
    "polysub",
    "polytrim",
    "polyval",
    "polyval2d",
    "polyval3d",
    "polyvalfromroots",
    "polyvander",
    "polyvander2d",
    "polyvander3d",
    "polyx",
    "polyzero",
]

polytrim = polyutils.trimcoef

polydomain = numpy.array([-1.0, 1.0])

polyzero = numpy.array([0])

polyone = numpy.array([1])

polyx = numpy.array([0, 1])


def polyline(off, scl):
    if scl != 0:
        return numpy.array([off, scl])
    else:
        return numpy.array([off])


def polyfromroots(roots):
    return polyutils._fromroots(polyline, polymul, roots)


def polyadd(c1, c2):
    return polyutils._add(c1, c2)


def polysub(c1, c2):
    return polyutils._sub(c1, c2)


def polymulx(c):
    [c] = polyutils.as_series([c])

    if len(c) == 1 and c[0] == 0:
        return c

    prd = numpy.empty(len(c) + 1, dtype=c.dtype)
    prd[0] = c[0] * 0
    prd[1:] = c
    return prd


def polymul(c1, c2):
    [c1, c2] = polyutils.as_series([c1, c2])
    ret = numpy.convolve(c1, c2)
    return polyutils.trimseq(ret)


def polydiv(c1, c2):
    [c1, c2] = polyutils.as_series([c1, c2])
    if c2[-1] == 0:
        raise ZeroDivisionError()

    lc1 = len(c1)
    lc2 = len(c2)
    if lc1 < lc2:
        return c1[:1] * 0, c1
    elif lc2 == 1:
        return c1 / c2[-1], c1[:1] * 0
    else:
        dlen = lc1 - lc2
        scl = c2[-1]
        c2 = c2[:-1] / scl
        i = dlen
        j = lc1 - 1
        while i >= 0:
            c1[i:j] -= c2 * c1[j]
            i -= 1
            j -= 1
        return c1[j + 1 :] / scl, polyutils.trimseq(c1[: j + 1])


def polypow(c, pow, maxpower=None):
    return polyutils._pow(numpy.convolve, c, pow, maxpower)


def polyder(c, m=1, scl=1, axis=0):
    c = numpy.array(c, ndmin=1, copy=True)
    if c.dtype.char in "?bBhHiIlLqQpP":
        c = c + 0.0
    cdt = c.dtype
    cnt = polyutils._as_int(m, "the order of derivation")
    iaxis = polyutils._as_int(axis, "the axis")
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


def polyint(c, m=1, k=None, lbnd=0, scl=1, axis=0):
    if k is None:
        k = []
    c = numpy.array(c, ndmin=1, copy=True)
    if c.dtype.char in "?bBhHiIlLqQpP":
        c = c + 0.0
    cdt = c.dtype
    if not numpy.iterable(k):
        k = [k]
    cnt = polyutils._as_int(m, "the order of integration")
    iaxis = polyutils._as_int(axis, "the axis")
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

    k = list(k) + [0] * (cnt - len(k))
    c = numpy.moveaxis(c, iaxis, 0)
    for i in range(cnt):
        n = len(c)
        c *= scl
        if n == 1 and numpy.all(c[0] == 0):
            c[0] += k[i]
        else:
            tmp = numpy.empty((n + 1,) + c.shape[1:], dtype=cdt)
            tmp[0] = c[0] * 0
            tmp[1] = c[0]
            for j in range(1, n):
                tmp[j + 1] = c[j] / (j + 1)
            tmp[0] += k[i] - polyval(lbnd, tmp)
            c = tmp
    c = numpy.moveaxis(c, 0, iaxis)
    return c


def polyval(x, c, tensor=True):
    c = numpy.array(c, ndmin=1)
    if c.dtype.char in "?bBhHiIlLqQpP":
        c = c + 0.0
    if isinstance(x, (tuple, list)):
        x = numpy.asarray(x)
    if isinstance(x, numpy.ndarray) and tensor:
        c = c.reshape(c.shape + (1,) * x.ndim)

    c0 = c[-1] + x * 0
    for i in range(2, len(c) + 1):
        c0 = c[-i] + c0 * x
    return c0


def polyvalfromroots(x, r, tensor=True):
    r = numpy.array(r, ndmin=1)
    if r.dtype.char in "?bBhHiIlLqQpP":
        r = r.astype(numpy.double)
    if isinstance(x, (tuple, list)):
        x = numpy.asarray(x)
    if isinstance(x, numpy.ndarray):
        if tensor:
            r = r.reshape(r.shape + (1,) * x.ndim)
        elif x.ndim >= r.ndim:
            raise ValueError("x.ndim must be < r.ndim when tensor == False")
    return numpy.prod(x - r, axis=0)


def polyval2d(x, y, c):
    return polyutils._valnd(polyval, c, x, y)


def polygrid2d(x, y, c):
    return polyutils._gridnd(polyval, c, x, y)


def polyval3d(x, y, z, c):
    return polyutils._valnd(polyval, c, x, y, z)


def polygrid3d(x, y, z, c):
    return polyutils._gridnd(polyval, c, x, y, z)


def polyvander(x, deg):
    ideg = polyutils._as_int(deg, "deg")
    if ideg < 0:
        raise ValueError("deg must be non-negative")

    x = numpy.array(x, ndmin=1) + 0.0
    dims = (ideg + 1,) + x.shape
    dtyp = x.dtype
    v = numpy.empty(dims, dtype=dtyp)
    v[0] = x * 0 + 1
    if ideg > 0:
        v[1] = x
        for i in range(2, ideg + 1):
            v[i] = v[i - 1] * x
    return numpy.moveaxis(v, 0, -1)


def polyvander2d(x, y, deg):
    return polyutils._vander_nd_flat((polyvander, polyvander), (x, y), deg)


def polyvander3d(x, y, z, deg):
    return polyutils._vander_nd_flat(
        (polyvander, polyvander, polyvander), (x, y, z), deg
    )


def polyfit(x, y, deg, rcond=None, full=False, w=None):
    return polyutils._fit(polyvander, x, y, deg, rcond, full, w)


def polycompanion(c):
    [c] = polyutils.as_series([c])
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


def polyroots(c):
    [c] = polyutils.as_series([c])
    if len(c) < 2:
        return numpy.array([], dtype=c.dtype)
    if len(c) == 2:
        return numpy.array([-c[0] / c[1]])

    m = polycompanion(c)[::-1, ::-1]
    r = numpy.linalg.eigvals(m)
    r.sort()
    return r


class Polynomial(ABCPolyBase):
    _add = staticmethod(polyadd)
    _sub = staticmethod(polysub)
    _mul = staticmethod(polymul)
    _div = staticmethod(polydiv)
    _pow = staticmethod(polypow)
    _val = staticmethod(polyval)
    _int = staticmethod(polyint)
    _der = staticmethod(polyder)
    _fit = staticmethod(polyfit)
    _line = staticmethod(polyline)
    _roots = staticmethod(polyroots)
    _fromroots = staticmethod(polyfromroots)

    domain = numpy.array(polydomain)
    window = numpy.array(polydomain)
    basis_name = None
