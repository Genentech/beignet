import numpy
import numpy.linalg

from ._normalize_axis_index import normalize_axis_index
from ._polybase import ABCPolyBase

# from .polynomial import (
#     Polynomial,
#     polyadd,
#     polycompanion,
#     polyder,
#     polydiv,
#     polydomain,
#     polyfit,
#     polyfromroots,
#     polygrid2d,
#     polygrid3d,
#     polyint,
#     polyline,
#     polymul,
#     polymulx,
#     polyone,
#     polypow,
#     polyroots,
#     polysub,
#     polytrim,
#     polyval,
#     polyval2d,
#     polyval3d,
#     polyvalfromroots,
#     polyvander,
#     polyvander2d,
#     polyvander3d,
#     polyx,
#     polyzero,
# )
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
    _vander_nd,
    _vander_nd_flat,
    as_series,
    getdomain,
    mapdomain,
    mapparms,
    trimcoef,
    trimseq,
)

polytrim = trimcoef


polydomain = numpy.array([-1, 1])


polyzero = numpy.array([0])


polyone = numpy.array([1])


polyx = numpy.array([0, 1])


def polyline(off, scl):
    if scl != 0:
        return numpy.array([off, scl])
    else:
        return numpy.array([off])


def polyfromroots(roots):
    return _fromroots(polyline, polymul, roots)


def polyadd(c1, c2):
    return _add(c1, c2)


def polysub(c1, c2):
    return _sub(c1, c2)


def polymulx(c):
    [c] = as_series([c])

    if len(c) == 1 and c[0] == 0:
        return c

    prd = numpy.empty(len(c) + 1, dtype=c.dtype)
    prd[0] = c[0] * 0
    prd[1:] = c
    return prd


def polymul(c1, c2):
    return trimseq(numpy.convolve(*as_series([c1, c2])))


def polydiv(c1, c2):
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
        dlen = lc1 - lc2
        scl = c2[-1]
        c2 = c2[:-1] / scl
        i = dlen
        j = lc1 - 1
        while i >= 0:
            c1[i:j] -= c2 * c1[j]
            i -= 1
            j -= 1
        return c1[j + 1 :] / scl, trimseq(c1[: j + 1])


def polypow(c, pow, maxpower=None):
    return _pow(numpy.convolve, c, pow, maxpower)


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


def polyint(c, m=1, k=None, lbnd=0, scl=1, axis=0):
    if k is None:
        k = []
    c = numpy.array(c, ndmin=1, copy=True)
    if c.dtype.char in "?bBhHiIlLqQpP":
        c = c + 0.0
    cdt = c.dtype
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
    c = numpy.array(c, ndmin=1, copy=False)
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
    r = numpy.array(r, ndmin=1, copy=False)
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
    return _valnd(polyval, c, x, y)


def polygrid2d(x, y, c):
    return _gridnd(polyval, c, x, y)


def polyval3d(x, y, z, c):
    return _valnd(polyval, c, x, y, z)


def polygrid3d(x, y, z, c):
    return _gridnd(polyval, c, x, y, z)


def polyvander(x, deg):
    ideg = _deprecate_as_int(deg, "deg")
    if ideg < 0:
        raise ValueError("deg must be non-negative")

    x = numpy.array(x, copy=False, ndmin=1) + 0.0
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
    return _vander_nd_flat((polyvander, polyvander), (x, y), deg)


def polyvander3d(x, y, z, deg):
    return _vander_nd_flat((polyvander, polyvander, polyvander), (x, y, z), deg)


def polyfit(x, y, deg, rcond=None, full=False, w=None):
    return _fit(polyvander, x, y, deg, rcond, full, w)


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


def polyroots(c):
    [c] = as_series([c])
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


chebtrim = trimcoef


def _cseries_to_zseries(c):
    n = c.size
    zs = numpy.zeros(2 * n - 1, dtype=c.dtype)
    zs[n - 1 :] = c / 2
    return zs + zs[::-1]


def _zseries_to_cseries(zs):
    n = (zs.size + 1) // 2
    c = zs[n - 1 :].copy()
    c[1:n] *= 2
    return c


def _zseries_mul(z1, z2):
    return numpy.convolve(z1, z2)


def _zseries_div(z1, z2):
    z1 = z1.copy()
    z2 = z2.copy()
    lc1 = len(z1)
    lc2 = len(z2)
    if lc2 == 1:
        z1 /= z2
        return z1, z1[:1] * 0
    elif lc1 < lc2:
        return z1[:1] * 0, z1
    else:
        dlen = lc1 - lc2
        scl = z2[0]
        z2 /= scl
        quo = numpy.empty(dlen + 1, dtype=z1.dtype)
        i = 0
        j = dlen
        while i < j:
            r = z1[i]
            quo[i] = z1[i]
            quo[dlen - i] = r
            tmp = r * z2
            z1[i : i + lc2] -= tmp
            z1[j : j + lc2] -= tmp
            i += 1
            j -= 1
        r = z1[i]
        quo[i] = r
        tmp = r * z2
        z1[i : i + lc2] -= tmp
        quo /= scl
        rem = z1[i + 1 : i - 1 + lc2].copy()
        return quo, rem


def _zseries_der(zs):
    n = len(zs) // 2
    ns = numpy.array([-1, 0, 1], dtype=zs.dtype)
    zs *= numpy.arange(-n, n + 1) * 2
    d, r = _zseries_div(zs, ns)
    return d


def _zseries_int(zs):
    n = 1 + len(zs) // 2
    ns = numpy.array([-1, 0, 1], dtype=zs.dtype)
    zs = _zseries_mul(zs, ns)
    div = numpy.arange(-n, n + 1) * 2
    zs[:n] /= div[:n]
    zs[n + 1 :] /= div[n + 1 :]
    zs[n] = 0
    return zs


def poly2cheb(pol):
    [pol] = as_series([pol])
    deg = len(pol) - 1
    res = 0
    for i in range(deg, -1, -1):
        res = chebadd(chebmulx(res), pol[i])
    return res


def cheb2poly(c):
    [c] = as_series([c])
    n = len(c)
    if n < 3:
        return c
    else:
        c0 = c[-2]
        c1 = c[-1]

        for i in range(n - 1, 1, -1):
            tmp = c0
            c0 = polysub(c[i - 2], c1)
            c1 = polyadd(tmp, polymulx(c1) * 2)
        return polyadd(c0, polymulx(c1))


chebdomain = numpy.array([-1, 1])


chebzero = numpy.array([0])


chebone = numpy.array([1])


chebx = numpy.array([0, 1])


def chebline(off, scl):
    if scl != 0:
        return numpy.array([off, scl])
    else:
        return numpy.array([off])


def chebfromroots(roots):
    return _fromroots(chebline, chebmul, roots)


def chebadd(c1, c2):
    return _add(c1, c2)


def chebsub(c1, c2):
    return _sub(c1, c2)


def chebmulx(c):
    [c] = as_series([c])

    if len(c) == 1 and c[0] == 0:
        return c

    prd = numpy.empty(len(c) + 1, dtype=c.dtype)
    prd[0] = c[0] * 0
    prd[1] = c[0]
    if len(c) > 1:
        tmp = c[1:] / 2
        prd[2:] = tmp
        prd[0:-2] += tmp
    return prd


def chebmul(c1, c2):
    [c1, c2] = as_series([c1, c2])
    z1 = _cseries_to_zseries(c1)
    z2 = _cseries_to_zseries(c2)
    prd = _zseries_mul(z1, z2)
    ret = _zseries_to_cseries(prd)
    return trimseq(ret)


def chebdiv(c1, c2):
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
        z1 = _cseries_to_zseries(c1)
        z2 = _cseries_to_zseries(c2)
        quo, rem = _zseries_div(z1, z2)
        quo = trimseq(_zseries_to_cseries(quo))
        rem = trimseq(_zseries_to_cseries(rem))
        return quo, rem


def chebpow(c, pow, maxpower=16):
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
        zs = _cseries_to_zseries(c)
        prd = zs
        for _ in range(2, power + 1):
            prd = numpy.convolve(prd, zs)
        return _zseries_to_cseries(prd)


def chebder(c, m=1, scl=1, axis=0):
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
            for j in range(n, 2, -1):
                der[j - 1] = (2 * j) * c[j]
                c[j - 2] += (j * c[j]) / (j - 2)
            if n > 1:
                der[1] = 4 * c[2]
            der[0] = c[1]
            c = der
    c = numpy.moveaxis(c, 0, iaxis)
    return c


def chebint(c, m=1, k=None, lbnd=0, scl=1, axis=0):
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
            tmp[1] = c[0]
            if n > 1:
                tmp[2] = c[1] / 4
            for j in range(2, n):
                tmp[j + 1] = c[j] / (2 * (j + 1))
                tmp[j - 1] -= c[j] / (2 * (j - 1))
            tmp[0] += k[i] - chebval(lbnd, tmp)
            c = tmp
    c = numpy.moveaxis(c, 0, iaxis)
    return c


def chebval(x, c, tensor=True):
    c = numpy.array(c, ndmin=1, copy=True)
    if c.dtype.char in "?bBhHiIlLqQpP":
        c = c.astype(numpy.double)
    if isinstance(x, (tuple, list)):
        x = numpy.asarray(x)
    if isinstance(x, numpy.ndarray) and tensor:
        c = c.reshape(c.shape + (1,) * x.ndim)

    if len(c) == 1:
        c0 = c[0]
        c1 = 0
    elif len(c) == 2:
        c0 = c[0]
        c1 = c[1]
    else:
        x2 = 2 * x
        c0 = c[-2]
        c1 = c[-1]
        for i in range(3, len(c) + 1):
            tmp = c0
            c0 = c[-i] - c1
            c1 = tmp + c1 * x2
    return c0 + c1 * x


def chebval2d(x, y, c):
    return _valnd(chebval, c, x, y)


def chebgrid2d(x, y, c):
    return _gridnd(chebval, c, x, y)


def chebval3d(x, y, z, c):
    return _valnd(chebval, c, x, y, z)


def chebgrid3d(x, y, z, c):
    return _gridnd(chebval, c, x, y, z)


def chebvander(x, deg):
    ideg = _deprecate_as_int(deg, "deg")
    if ideg < 0:
        raise ValueError("deg must be non-negative")

    x = numpy.array(x, copy=False, ndmin=1) + 0.0
    dims = (ideg + 1,) + x.shape
    dtyp = x.dtype
    v = numpy.empty(dims, dtype=dtyp)

    v[0] = x * 0 + 1
    if ideg > 0:
        x2 = 2 * x
        v[1] = x
        for i in range(2, ideg + 1):
            v[i] = v[i - 1] * x2 - v[i - 2]
    return numpy.moveaxis(v, 0, -1)


def chebvander2d(x, y, deg):
    return _vander_nd_flat((chebvander, chebvander), (x, y), deg)


def chebvander3d(x, y, z, deg):
    return _vander_nd_flat((chebvander, chebvander, chebvander), (x, y, z), deg)


def chebfit(x, y, deg, rcond=None, full=False, w=None):
    return _fit(chebvander, x, y, deg, rcond, full, w)


def chebcompanion(c):
    [c] = as_series([c])
    if len(c) < 2:
        raise ValueError("Series must have maximum degree of at least 1.")
    if len(c) == 2:
        return numpy.array([[-c[0] / c[1]]])

    n = len(c) - 1
    mat = numpy.zeros((n, n), dtype=c.dtype)
    scl = numpy.array([1.0] + [numpy.sqrt(0.5)] * (n - 1))
    top = mat.reshape(-1)[1 :: n + 1]
    bot = mat.reshape(-1)[n :: n + 1]
    top[0] = numpy.sqrt(0.5)
    top[1:] = 1 / 2
    bot[...] = top
    mat[:, -1] -= (c[:-1] / c[-1]) * (scl / scl[-1]) * 0.5
    return mat


def chebroots(c):
    [c] = as_series([c])
    if len(c) < 2:
        return numpy.array([], dtype=c.dtype)
    if len(c) == 2:
        return numpy.array([-c[0] / c[1]])

    m = chebcompanion(c)[::-1, ::-1]
    r = numpy.linalg.eigvals(m)
    r.sort()
    return r


def chebinterpolate(func, deg, args=()):
    deg = numpy.asarray(deg)

    if deg.ndim > 0 or deg.dtype.kind not in "iu" or deg.size == 0:
        raise TypeError("deg must be an int")
    if deg < 0:
        raise ValueError("expected deg >= 0")

    order = deg + 1
    xcheb = chebpts1(order)
    yfunc = func(xcheb, *args)
    m = chebvander(xcheb, deg)
    c = numpy.dot(m.T, yfunc)
    c[0] /= order
    c[1:] /= 0.5 * order

    return c


def chebgauss(deg):
    ideg = _deprecate_as_int(deg, "deg")
    if ideg <= 0:
        raise ValueError("deg must be a positive integer")

    x = numpy.cos(numpy.pi * numpy.arange(1, 2 * ideg, 2) / (2.0 * ideg))
    w = numpy.ones(ideg) * (numpy.pi / ideg)

    return x, w


def chebweight(x):
    return 1.0 / (numpy.sqrt(1.0 + x) * numpy.sqrt(1.0 - x))


def chebpts1(npts):
    _npts = int(npts)
    if _npts != npts:
        raise ValueError("npts must be integer")
    if _npts < 1:
        raise ValueError("npts must be >= 1")

    x = 0.5 * numpy.pi / _npts * numpy.arange(-_npts + 1, _npts + 1, 2)
    return numpy.sin(x)


def chebpts2(npts):
    _npts = int(npts)
    if _npts != npts:
        raise ValueError("npts must be integer")
    if _npts < 2:
        raise ValueError("npts must be >= 2")

    x = numpy.linspace(-numpy.pi, 0, _npts)
    return numpy.cos(x)


class Chebyshev(ABCPolyBase):
    _add = staticmethod(chebadd)
    _sub = staticmethod(chebsub)
    _mul = staticmethod(chebmul)
    _div = staticmethod(chebdiv)
    _pow = staticmethod(chebpow)
    _val = staticmethod(chebval)
    _int = staticmethod(chebint)
    _der = staticmethod(chebder)
    _fit = staticmethod(chebfit)
    _line = staticmethod(chebline)
    _roots = staticmethod(chebroots)
    _fromroots = staticmethod(chebfromroots)

    @classmethod
    def interpolate(cls, func, deg, domain=None, args=()):
        if domain is None:
            domain = cls.domain

        def xfunc(x):
            return func(mapdomain(x, cls.window, domain), *args)

        coef = chebinterpolate(xfunc, deg)
        return cls(coef, domain=domain)

    domain = numpy.array(chebdomain)
    window = numpy.array(chebdomain)
    basis_name = "T"


hermtrim = trimcoef


def poly2herm(pol):
    [pol] = as_series([pol])
    deg = len(pol) - 1
    res = 0
    for i in range(deg, -1, -1):
        res = hermadd(hermmulx(res), pol[i])
    return res


def herm2poly(c):
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


hermetrim = trimcoef


def poly2herme(pol):
    [pol] = as_series([pol])
    deg = len(pol) - 1
    res = 0
    for i in range(deg, -1, -1):
        res = hermeadd(hermemulx(res), pol[i])
    return res


def herme2poly(c):
    [c] = as_series([c])
    n = len(c)
    if n == 1:
        return c
    if n == 2:
        return c
    else:
        c0 = c[-2]
        c1 = c[-1]

        for i in range(n - 1, 1, -1):
            tmp = c0
            c0 = polysub(c[i - 2], c1 * (i - 1))
            c1 = polyadd(tmp, polymulx(c1))
        return polyadd(c0, polymulx(c1))


hermedomain = numpy.array([-1, 1])


hermezero = numpy.array([0])


hermeone = numpy.array([1])


hermex = numpy.array([0, 1])


def hermeline(off, scl):
    if scl != 0:
        return numpy.array([off, scl])
    else:
        return numpy.array([off])


def hermefromroots(roots):
    return _fromroots(hermeline, hermemul, roots)


def hermeadd(c1, c2):
    return _add(c1, c2)


def hermesub(c1, c2):
    return _sub(c1, c2)


def hermemulx(c):
    [c] = as_series([c])

    if len(c) == 1 and c[0] == 0:
        return c

    prd = numpy.empty(len(c) + 1, dtype=c.dtype)
    prd[0] = c[0] * 0
    prd[1] = c[0]
    for i in range(1, len(c)):
        prd[i + 1] = c[i]
        prd[i - 1] += c[i] * i
    return prd


def hermemul(c1, c2):
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
            c0 = hermesub(c[-i] * xs, c1 * (nd - 1))
            c1 = hermeadd(tmp, hermemulx(c1))
    return hermeadd(c0, hermemulx(c1))


def hermediv(c1, c2):
    return _div(hermemul, c1, c2)


def hermepow(c, pow, maxpower=16):
    return _pow(hermemul, c, pow, maxpower)


def hermeder(c, m=1, scl=1, axis=0):
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
        return c[:1] * 0
    else:
        for _ in range(cnt):
            n = n - 1
            c *= scl
            der = numpy.empty((n,) + c.shape[1:], dtype=c.dtype)
            for j in range(n, 0, -1):
                der[j - 1] = j * c[j]
            c = der
    c = numpy.moveaxis(c, 0, iaxis)
    return c


def hermeint(c, m=1, k=None, lbnd=0, scl=1, axis=0):
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
            tmp[1] = c[0]
            for j in range(1, n):
                tmp[j + 1] = c[j] / (j + 1)
            tmp[0] += k[i] - hermeval(lbnd, tmp)
            c = tmp
    c = numpy.moveaxis(c, 0, iaxis)
    return c


def hermeval(x, c, tensor=True):
    c = numpy.array(c, ndmin=1, copy=False)
    if c.dtype.char in "?bBhHiIlLqQpP":
        c = c.astype(numpy.double)
    if isinstance(x, (tuple, list)):
        x = numpy.asarray(x)
    if isinstance(x, numpy.ndarray) and tensor:
        c = c.reshape(c.shape + (1,) * x.ndim)

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
            c0 = c[-i] - c1 * (nd - 1)
            c1 = tmp + c1 * x
    return c0 + c1 * x


def hermeval2d(x, y, c):
    return _valnd(hermeval, c, x, y)


def hermegrid2d(x, y, c):
    return _gridnd(hermeval, c, x, y)


def hermeval3d(x, y, z, c):
    return _valnd(hermeval, c, x, y, z)


def hermegrid3d(x, y, z, c):
    return _gridnd(hermeval, c, x, y, z)


def hermevander(x, deg):
    ideg = _deprecate_as_int(deg, "deg")
    if ideg < 0:
        raise ValueError("deg must be non-negative")

    x = numpy.array(x, copy=False, ndmin=1) + 0.0
    dims = (ideg + 1,) + x.shape
    dtyp = x.dtype
    v = numpy.empty(dims, dtype=dtyp)
    v[0] = x * 0 + 1
    if ideg > 0:
        v[1] = x
        for i in range(2, ideg + 1):
            v[i] = v[i - 1] * x - v[i - 2] * (i - 1)
    return numpy.moveaxis(v, 0, -1)


def hermevander2d(x, y, deg):
    return _vander_nd_flat((hermevander, hermevander), (x, y), deg)


def hermevander3d(x, y, z, deg):
    return _vander_nd_flat((hermevander, hermevander, hermevander), (x, y, z), deg)


def hermefit(x, y, deg, rcond=None, full=False, w=None):
    return _fit(hermevander, x, y, deg, rcond, full, w)


def hermecompanion(c):
    [c] = as_series([c])
    if len(c) < 2:
        raise ValueError("Series must have maximum degree of at least 1.")
    if len(c) == 2:
        return numpy.array([[-c[0] / c[1]]])

    n = len(c) - 1
    mat = numpy.zeros((n, n), dtype=c.dtype)
    scl = numpy.hstack((1.0, 1.0 / numpy.sqrt(numpy.arange(n - 1, 0, -1))))
    scl = numpy.multiply.accumulate(scl)[::-1]
    top = mat.reshape(-1)[1 :: n + 1]
    bot = mat.reshape(-1)[n :: n + 1]
    top[...] = numpy.sqrt(numpy.arange(1, n))
    bot[...] = top
    mat[:, -1] -= scl * c[:-1] / c[-1]
    return mat


def hermeroots(c):
    [c] = as_series([c])
    if len(c) <= 1:
        return numpy.array([], dtype=c.dtype)
    if len(c) == 2:
        return numpy.array([-c[0] / c[1]])

    m = hermecompanion(c)[::-1, ::-1]
    r = numpy.linalg.eigvals(m)
    r.sort()
    return r


def _normed_hermite_e_n(x, n):
    if n == 0:
        return numpy.full(x.shape, 1 / numpy.sqrt(numpy.sqrt(2 * numpy.pi)))

    c0 = 0.0
    c1 = 1.0 / numpy.sqrt(numpy.sqrt(2 * numpy.pi))
    nd = float(n)
    for _ in range(n - 1):
        tmp = c0
        c0 = -c1 * numpy.sqrt((nd - 1.0) / nd)
        c1 = tmp + c1 * x * numpy.sqrt(1.0 / nd)
        nd = nd - 1.0
    return c0 + c1 * x


def hermegauss(deg):
    ideg = _deprecate_as_int(deg, "deg")
    if ideg <= 0:
        raise ValueError("deg must be a positive integer")

    c = numpy.array([0] * deg + [1])
    m = hermecompanion(c)
    x = numpy.linalg.eigvalsh(m)

    dy = _normed_hermite_e_n(x, ideg)
    df = _normed_hermite_e_n(x, ideg - 1) * numpy.sqrt(ideg)
    x -= dy / df

    fm = _normed_hermite_e_n(x, ideg - 1)
    fm /= numpy.abs(fm).max()
    w = 1 / (fm * fm)

    w = (w + w[::-1]) / 2
    x = (x - x[::-1]) / 2

    w *= numpy.sqrt(2 * numpy.pi) / w.sum()

    return x, w


def hermeweight(x):
    return numpy.exp(-0.5 * x**2)


class HermiteE(ABCPolyBase):
    _add = staticmethod(hermeadd)
    _sub = staticmethod(hermesub)
    _mul = staticmethod(hermemul)
    _div = staticmethod(hermediv)
    _pow = staticmethod(hermepow)
    _val = staticmethod(hermeval)
    _int = staticmethod(hermeint)
    _der = staticmethod(hermeder)
    _fit = staticmethod(hermefit)
    _line = staticmethod(hermeline)
    _roots = staticmethod(hermeroots)
    _fromroots = staticmethod(hermefromroots)

    domain = numpy.array(hermedomain)
    window = numpy.array(hermedomain)
    basis_name = "He"


lagtrim = trimcoef


def poly2lag(pol):
    [pol] = as_series([pol])
    res = 0
    for p in pol[::-1]:
        res = lagadd(lagmulx(res), p)
    return res


def lag2poly(c):
    [c] = as_series([c])
    n = len(c)
    if n == 1:
        return c
    else:
        c0 = c[-2]
        c1 = c[-1]

        for i in range(n - 1, 1, -1):
            tmp = c0
            c0 = polysub(c[i - 2], (c1 * (i - 1)) / i)
            c1 = polyadd(tmp, polysub((2 * i - 1) * c1, polymulx(c1)) / i)
        return polyadd(c0, polysub(c1, polymulx(c1)))


lagdomain = numpy.array([0, 1])


lagzero = numpy.array([0])


lagone = numpy.array([1])


lagx = numpy.array([1, -1])


def lagline(off, scl):
    if scl != 0:
        return numpy.array([off + scl, -scl])
    else:
        return numpy.array([off])


def lagfromroots(roots):
    return _fromroots(lagline, lagmul, roots)


def lagadd(c1, c2):
    return _add(c1, c2)


def lagsub(c1, c2):
    return _sub(c1, c2)


def lagmulx(c):
    [c] = as_series([c])

    if len(c) == 1 and c[0] == 0:
        return c

    prd = numpy.empty(len(c) + 1, dtype=c.dtype)
    prd[0] = c[0]
    prd[1] = -c[0]
    for i in range(1, len(c)):
        prd[i + 1] = -c[i] * (i + 1)
        prd[i] += c[i] * (2 * i + 1)
        prd[i - 1] -= c[i] * i
    return prd


def lagmul(c1, c2):
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
            c0 = lagsub(c[-i] * xs, (c1 * (nd - 1)) / nd)
            c1 = lagadd(tmp, lagsub((2 * nd - 1) * c1, lagmulx(c1)) / nd)
    return lagadd(c0, lagsub(c1, lagmulx(c1)))


def lagdiv(c1, c2):
    return _div(lagmul, c1, c2)


def lagpow(c, pow, maxpower=16):
    return _pow(lagmul, c, pow, maxpower)


def lagder(c, m=1, scl=1, axis=0):
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
            for j in range(n, 1, -1):
                der[j - 1] = -c[j]
                c[j - 1] += c[j]
            der[0] = -c[1]
            c = der
    c = numpy.moveaxis(c, 0, iaxis)
    return c


def lagint(c, m=1, k=None, lbnd=0, scl=1, axis=0):
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
            tmp[0] = c[0]
            tmp[1] = -c[0]
            for j in range(1, n):
                tmp[j] += c[j]
                tmp[j + 1] = -c[j]
            tmp[0] += k[i] - lagval(lbnd, tmp)
            c = tmp
    c = numpy.moveaxis(c, 0, iaxis)
    return c


def lagval(x, c, tensor=True):
    c = numpy.array(c, ndmin=1, copy=False)
    if c.dtype.char in "?bBhHiIlLqQpP":
        c = c.astype(numpy.double)
    if isinstance(x, (tuple, list)):
        x = numpy.asarray(x)
    if isinstance(x, numpy.ndarray) and tensor:
        c = c.reshape(c.shape + (1,) * x.ndim)

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
            c0 = c[-i] - (c1 * (nd - 1)) / nd
            c1 = tmp + (c1 * ((2 * nd - 1) - x)) / nd
    return c0 + c1 * (1 - x)


def lagval2d(x, y, c):
    return _valnd(lagval, c, x, y)


def laggrid2d(x, y, c):
    return _gridnd(lagval, c, x, y)


def lagval3d(x, y, z, c):
    return _valnd(lagval, c, x, y, z)


def laggrid3d(x, y, z, c):
    return _gridnd(lagval, c, x, y, z)


def lagvander(x, deg):
    ideg = _deprecate_as_int(deg, "deg")
    if ideg < 0:
        raise ValueError("deg must be non-negative")

    x = numpy.array(x, copy=False, ndmin=1) + 0.0
    dims = (ideg + 1,) + x.shape
    dtyp = x.dtype
    v = numpy.empty(dims, dtype=dtyp)
    v[0] = x * 0 + 1
    if ideg > 0:
        v[1] = 1 - x
        for i in range(2, ideg + 1):
            v[i] = (v[i - 1] * (2 * i - 1 - x) - v[i - 2] * (i - 1)) / i
    return numpy.moveaxis(v, 0, -1)


def lagvander2d(x, y, deg):
    return _vander_nd_flat((lagvander, lagvander), (x, y), deg)


def lagvander3d(x, y, z, deg):
    return _vander_nd_flat((lagvander, lagvander, lagvander), (x, y, z), deg)


def lagfit(x, y, deg, rcond=None, full=False, w=None):
    return _fit(lagvander, x, y, deg, rcond, full, w)


def lagcompanion(c):
    [c] = as_series([c])
    if len(c) < 2:
        raise ValueError("Series must have maximum degree of at least 1.")
    if len(c) == 2:
        return numpy.array([[1 + c[0] / c[1]]])

    n = len(c) - 1
    mat = numpy.zeros((n, n), dtype=c.dtype)
    top = mat.reshape(-1)[1 :: n + 1]
    mid = mat.reshape(-1)[0 :: n + 1]
    bot = mat.reshape(-1)[n :: n + 1]
    top[...] = -numpy.arange(1, n)
    mid[...] = 2.0 * numpy.arange(n) + 1.0
    bot[...] = top
    mat[:, -1] += (c[:-1] / c[-1]) * n
    return mat


def lagroots(c):
    [c] = as_series([c])
    if len(c) <= 1:
        return numpy.array([], dtype=c.dtype)
    if len(c) == 2:
        return numpy.array([1 + c[0] / c[1]])

    m = lagcompanion(c)[::-1, ::-1]
    r = numpy.linalg.eigvals(m)
    r.sort()
    return r


def laggauss(deg):
    ideg = _deprecate_as_int(deg, "deg")
    if ideg <= 0:
        raise ValueError("deg must be a positive integer")

    c = numpy.array([0] * deg + [1])
    m = lagcompanion(c)
    x = numpy.linalg.eigvalsh(m)

    dy = lagval(x, c)
    df = lagval(x, lagder(c))
    x -= dy / df

    fm = lagval(x, c[1:])
    fm /= numpy.abs(fm).max()
    df /= numpy.abs(df).max()
    w = 1 / (fm * df)

    w /= w.sum()

    return x, w


def lagweight(x):
    w = numpy.exp(-x)
    return w


class Laguerre(ABCPolyBase):
    _add = staticmethod(lagadd)
    _sub = staticmethod(lagsub)
    _mul = staticmethod(lagmul)
    _div = staticmethod(lagdiv)
    _pow = staticmethod(lagpow)
    _val = staticmethod(lagval)
    _int = staticmethod(lagint)
    _der = staticmethod(lagder)
    _fit = staticmethod(lagfit)
    _line = staticmethod(lagline)
    _roots = staticmethod(lagroots)
    _fromroots = staticmethod(lagfromroots)

    domain = numpy.array(lagdomain)
    window = numpy.array(lagdomain)
    basis_name = "L"


legtrim = trimcoef


def poly2leg(pol):
    [pol] = as_series([pol])
    deg = len(pol) - 1
    res = 0
    for i in range(deg, -1, -1):
        res = legadd(legmulx(res), pol[i])
    return res


def leg2poly(c):
    [c] = as_series([c])
    n = len(c)
    if n < 3:
        return c
    else:
        c0 = c[-2]
        c1 = c[-1]

        for i in range(n - 1, 1, -1):
            tmp = c0
            c0 = polysub(c[i - 2], (c1 * (i - 1)) / i)
            c1 = polyadd(tmp, (polymulx(c1) * (2 * i - 1)) / i)
        return polyadd(c0, polymulx(c1))


legdomain = numpy.array([-1, 1])


legzero = numpy.array([0])


legone = numpy.array([1])


legx = numpy.array([0, 1])


def legline(off, scl):
    if scl != 0:
        return numpy.array([off, scl])
    else:
        return numpy.array([off])


def legfromroots(roots):
    return _fromroots(legline, legmul, roots)


def legadd(c1, c2):
    return _add(c1, c2)


def legsub(c1, c2):
    return _sub(c1, c2)


def legmulx(c):
    [c] = as_series([c])

    if len(c) == 1 and c[0] == 0:
        return c

    prd = numpy.empty(len(c) + 1, dtype=c.dtype)
    prd[0] = c[0] * 0
    prd[1] = c[0]
    for i in range(1, len(c)):
        j = i + 1
        k = i - 1
        s = i + j
        prd[j] = (c[i] * j) / s
        prd[k] += (c[i] * i) / s
    return prd


def legmul(c1, c2):
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
            c0 = legsub(c[-i] * xs, (c1 * (nd - 1)) / nd)
            c1 = legadd(tmp, (legmulx(c1) * (2 * nd - 1)) / nd)
    return legadd(c0, legmulx(c1))


def legdiv(c1, c2):
    return _div(legmul, c1, c2)


def legpow(c, pow, maxpower=16):
    return _pow(legmul, c, pow, maxpower)


def legder(c, m=1, scl=1, axis=0):
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
            for j in range(n, 2, -1):
                der[j - 1] = (2 * j - 1) * c[j]
                c[j - 2] += c[j]
            if n > 1:
                der[1] = 3 * c[2]
            der[0] = c[1]
            c = der
    c = numpy.moveaxis(c, 0, iaxis)
    return c


def legint(c, m=1, k=None, lbnd=0, scl=1, axis=0):
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
            tmp[1] = c[0]
            if n > 1:
                tmp[2] = c[1] / 3
            for j in range(2, n):
                t = c[j] / (2 * j + 1)
                tmp[j + 1] = t
                tmp[j - 1] -= t
            tmp[0] += k[i] - legval(lbnd, tmp)
            c = tmp
    c = numpy.moveaxis(c, 0, iaxis)
    return c


def legval(x, c, tensor=True):
    c = numpy.array(c, ndmin=1, copy=False)
    if c.dtype.char in "?bBhHiIlLqQpP":
        c = c.astype(numpy.double)
    if isinstance(x, (tuple, list)):
        x = numpy.asarray(x)
    if isinstance(x, numpy.ndarray) and tensor:
        c = c.reshape(c.shape + (1,) * x.ndim)

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
            c0 = c[-i] - (c1 * (nd - 1)) / nd
            c1 = tmp + (c1 * x * (2 * nd - 1)) / nd
    return c0 + c1 * x


def legval2d(x, y, c):
    return _valnd(legval, c, x, y)


def leggrid2d(x, y, c):
    return _gridnd(legval, c, x, y)


def legval3d(x, y, z, c):
    return _valnd(legval, c, x, y, z)


def leggrid3d(x, y, z, c):
    return _gridnd(legval, c, x, y, z)


def legvander(x, deg):
    ideg = _deprecate_as_int(deg, "deg")
    if ideg < 0:
        raise ValueError("deg must be non-negative")

    x = numpy.array(x, copy=False, ndmin=1) + 0.0
    dims = (ideg + 1,) + x.shape
    dtyp = x.dtype
    v = numpy.empty(dims, dtype=dtyp)

    v[0] = x * 0 + 1
    if ideg > 0:
        v[1] = x
        for i in range(2, ideg + 1):
            v[i] = (v[i - 1] * x * (2 * i - 1) - v[i - 2] * (i - 1)) / i
    return numpy.moveaxis(v, 0, -1)


def legvander2d(x, y, deg):
    return _vander_nd_flat((legvander, legvander), (x, y), deg)


def legvander3d(x, y, z, deg):
    return _vander_nd_flat((legvander, legvander, legvander), (x, y, z), deg)


def legfit(x, y, deg, rcond=None, full=False, w=None):
    return _fit(legvander, x, y, deg, rcond, full, w)


def legcompanion(c):
    [c] = as_series([c])
    if len(c) < 2:
        raise ValueError("Series must have maximum degree of at least 1.")
    if len(c) == 2:
        return numpy.array([[-c[0] / c[1]]])

    n = len(c) - 1
    mat = numpy.zeros((n, n), dtype=c.dtype)
    scl = 1.0 / numpy.sqrt(2 * numpy.arange(n) + 1)
    top = mat.reshape(-1)[1 :: n + 1]
    bot = mat.reshape(-1)[n :: n + 1]
    top[...] = numpy.arange(1, n) * scl[: n - 1] * scl[1:n]
    bot[...] = top
    mat[:, -1] -= (c[:-1] / c[-1]) * (scl / scl[-1]) * (n / (2 * n - 1))
    return mat


def legroots(c):
    [c] = as_series([c])
    if len(c) < 2:
        return numpy.array([], dtype=c.dtype)
    if len(c) == 2:
        return numpy.array([-c[0] / c[1]])

    m = legcompanion(c)[::-1, ::-1]
    r = numpy.linalg.eigvals(m)
    r.sort()
    return r


def leggauss(deg):
    ideg = _deprecate_as_int(deg, "deg")
    if ideg <= 0:
        raise ValueError("deg must be a positive integer")

    c = numpy.array([0] * deg + [1])
    m = legcompanion(c)
    x = numpy.linalg.eigvalsh(m)

    dy = legval(x, c)
    df = legval(x, legder(c))
    x -= dy / df

    fm = legval(x, c[1:])
    fm /= numpy.abs(fm).max()
    df /= numpy.abs(df).max()
    w = 1 / (fm * df)

    w = (w + w[::-1]) / 2
    x = (x - x[::-1]) / 2

    w *= 2.0 / w.sum()

    return x, w


def legweight(x):
    return x * 0.0 + 1.0


class Legendre(ABCPolyBase):
    _add = staticmethod(legadd)
    _sub = staticmethod(legsub)
    _mul = staticmethod(legmul)
    _div = staticmethod(legdiv)
    _pow = staticmethod(legpow)
    _val = staticmethod(legval)
    _int = staticmethod(legint)
    _der = staticmethod(legder)
    _fit = staticmethod(legfit)
    _line = staticmethod(legline)
    _roots = staticmethod(legroots)
    _fromroots = staticmethod(legfromroots)

    domain = numpy.array(legdomain)
    window = numpy.array(legdomain)
    basis_name = "P"
