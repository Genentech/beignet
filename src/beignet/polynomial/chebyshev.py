import numpy
import numpy.linalg

from ._normalize_axis_index import normalize_axis_index
from ._polybase import ABCPolyBase
from .polyutils import (
    _add,
    _deprecate_as_int,
    _fit,
    _fromroots,
    _gridnd,
    _sub,
    _valnd,
    _vander_nd_flat,
    as_series,
    mapdomain,
    trimcoef,
    trimseq,
)

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
    from .polynomial import polyadd, polymulx, polysub

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
