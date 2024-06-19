import operator

import numpy
import numpy.linalg
import torch

from .__add import _add
from .__as_series import _as_series
from .__c_series_to_z_series import _c_series_to_z_series
from .__div import _div
from .__evaluate import _evaluate
from .__fit import _fit
from .__from_roots import _from_roots
from .__grid import _grid
from .__map_parameters import _map_parameters
from .__normalize_axis_index import _normalize_axis_index
from .__normed_hermite_e_n import _normed_hermite_e_n
from .__normed_hermite_n import _normed_hermite_n
from .__pow import _pow
from .__sub import _sub
from .__trim_coefficients import _trim_coefficients
from .__trim_sequence import _trim_sequence
from .__vander_nd_flat import _vander_nd_flat
from .__z_series_to_c_series import _z_series_to_c_series
from .__zseries_div import _zseries_div
from ._hermadd import hermadd
from ._hermeadd import hermeadd
from ._hermesub import hermesub
from ._hermline import hermline
from ._hermsub import hermsub
from ._lagadd import lagadd
from ._lagline import lagline
from ._lagsub import lagsub
from ._legadd import legadd
from ._legline import legline
from ._legsub import legsub
from ._polyadd import polyadd
from ._polyline import polyline
from ._polysub import polysub


class RankWarning(RuntimeWarning):
    pass


chebtrim = _trim_coefficients
hermetrim = _trim_coefficients
hermtrim = _trim_coefficients
lagtrim = _trim_coefficients
legtrim = _trim_coefficients
polytrim = _trim_coefficients


chebdomain = torch.tensor([-1.0, 1.0])

chebone = torch.tensor([1])

chebx = torch.tensor([0, 1])

chebzero = torch.tensor([0])


def cheb2poly(input):
    [input] = _as_series([input])

    n = len(input)
    if n < 3:
        return input
    else:
        c0 = input[-2]
        c1 = input[-1]

        for i in range(n - 1, 1, -1):
            tmp = c0
            c0 = polysub(input[i - 2], c1)
            c1 = polyadd(tmp, polymulx(c1) * 2)
        return polyadd(c0, polymulx(c1))


def chebcompanion(c):
    [c] = _as_series([c])
    if len(c) < 2:
        raise ValueError("Series must have maximum degree of at least 1.")
    if len(c) == 2:
        return torch.tensor([[-c[0] / c[1]]])

    n = len(c) - 1
    mat = numpy.zeros((n, n), dtype=c.dtype)
    scl = torch.tensor([1.0] + [numpy.sqrt(0.5)] * (n - 1))
    top = mat.reshape(-1)[1 :: n + 1]
    bot = mat.reshape(-1)[n :: n + 1]
    top[0] = numpy.sqrt(0.5)
    top[1:] = 1 / 2
    bot[...] = top
    mat[:, -1] -= (c[:-1] / c[-1]) * (scl / scl[-1]) * 0.5
    return mat


def chebder(c, m=1, scl=1, axis=0):
    c = numpy.array(c, ndmin=1)
    if c.dtype.char in "?bBhHiIlLqQpP":
        c = c.astype(numpy.double)
    cnt = operator.index(m)
    iaxis = operator.index(axis)
    if cnt < 0:
        raise ValueError("The order of derivation must be non-negative")
    iaxis = _normalize_axis_index(iaxis, c.ndim)

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


def chebdiv(c1, c2):
    [c1, c2] = _as_series([c1, c2])
    if c2[-1] == 0:
        raise ZeroDivisionError()

    lc1 = len(c1)
    lc2 = len(c2)

    if lc1 < lc2:
        return c1[:1] * 0, c1
    elif lc2 == 1:
        return c1 / c2[-1], c1[:1] * 0
    else:
        z1 = _c_series_to_z_series(c1)
        z2 = _c_series_to_z_series(c2)

        quo, rem = _zseries_div(z1, z2)

        quo = _trim_sequence(_z_series_to_c_series(quo))
        rem = _trim_sequence(_z_series_to_c_series(rem))

        return quo, rem


def chebfit(x, y, deg, rcond=None, full=False, w=None):
    return _fit(chebvander, x, y, deg, rcond, full, w)


def chebfromroots(input):
    return _from_roots(chebline, chebmul, input)


def chebgauss(input):
    ideg = operator.index(input)

    if ideg <= 0:
        raise ValueError("deg must be a positive integer")

    output = numpy.arange(1, 2 * ideg, 2) / (2.0 * ideg)

    output = output * numpy.pi

    output = numpy.cos(output)

    weight = numpy.ones(ideg) * (numpy.pi / ideg)

    return output, weight


def chebgrid2d(x, y, c):
    return _grid(chebval, c, x, y)


def chebgrid3d(x, y, z, c):
    return _grid(chebval, c, x, y, z)


def chebint(c, m=1, k=None, lbnd=0, scl=1, axis=0):
    if k is None:
        k = []
    c = numpy.array(c, ndmin=1)
    if c.dtype.char in "?bBhHiIlLqQpP":
        c = c.astype(numpy.double)
    if not numpy.iterable(k):
        k = [k]
    cnt = operator.index(m)
    iaxis = operator.index(axis)
    if cnt < 0:
        raise ValueError("The order of integration must be non-negative")
    if len(k) > cnt:
        raise ValueError("Too many integration constants")
    if numpy.ndim(lbnd) != 0:
        raise ValueError("lbnd must be a scalar.")
    if numpy.ndim(scl) != 0:
        raise ValueError("scl must be a scalar.")
    iaxis = _normalize_axis_index(iaxis, c.ndim)

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


def chebline(input, other):
    if other != 0:
        return torch.tensor([input, other])
    else:
        return torch.tensor([input])


def chebmul(input, other):
    [input, other] = _as_series([input, other])

    input = _c_series_to_z_series(input)
    other = _c_series_to_z_series(other)

    output = numpy.convolve(input, other)

    n = (output.size + 1) // 2

    output = output[n - 1 :]

    output[1:n] = output[1:n] * 2

    if len(output) != 0 and output[-1] == 0:
        for index in range(len(output) - 1, -1, -1):
            if output[index] != 0:
                break

        output = output[: index + 1]

    return output


def chebmulx(c):
    [c] = _as_series([c])

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


def chebpow(c, pow, maxpower=16):
    [c] = _as_series([c])
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
        zs = _c_series_to_z_series(c)
        prd = zs
        for _ in range(2, power + 1):
            prd = numpy.convolve(prd, zs)
        return _z_series_to_c_series(prd)


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


def chebroots(c):
    [c] = _as_series([c])
    if len(c) < 2:
        return torch.tensor([], dtype=c.dtype)
    if len(c) == 2:
        return torch.tensor([-c[0] / c[1]])

    m = chebcompanion(c)[::-1, ::-1]
    r = numpy.linalg.eigvals(m)
    r.sort()
    return r


def chebval(x, c, tensor=True):
    c = numpy.array(c, ndmin=1)
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
    return _evaluate(chebval, c, x, y)


def chebval3d(x, y, z, c):
    return _evaluate(chebval, c, x, y, z)


def chebvander(x, deg):
    ideg = operator.index(deg)
    if ideg < 0:
        raise ValueError("deg must be non-negative")

    x = numpy.array(x, ndmin=1) + 0.0
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


def chebweight(x):
    w = 1.0 / (numpy.sqrt(1.0 + x) * numpy.sqrt(1.0 - x))
    return w


def herm2poly(input):
    [input] = _as_series([input])
    n = len(input)
    if n == 1:
        return input
    if n == 2:
        input[1] *= 2
        return input
    else:
        c0 = input[-2]
        c1 = input[-1]

        for i in range(n - 1, 1, -1):
            tmp = c0
            c0 = polysub(input[i - 2], c1 * (2 * (i - 1)))
            c1 = polyadd(tmp, polymulx(c1) * 2)
        return polyadd(c0, polymulx(c1) * 2)


def hermcompanion(c):
    [c] = _as_series([c])
    if len(c) < 2:
        raise ValueError("Series must have maximum degree of at least 1.")
    if len(c) == 2:
        return torch.tensor([[-0.5 * c[0] / c[1]]])

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


def hermder(c, m=1, scl=1, axis=0):
    c = numpy.array(c, ndmin=1)
    if c.dtype.char in "?bBhHiIlLqQpP":
        c = c.astype(numpy.double)
    cnt = operator.index(m)
    iaxis = operator.index(axis)
    if cnt < 0:
        raise ValueError("The order of derivation must be non-negative")
    iaxis = _normalize_axis_index(iaxis, c.ndim)

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


def hermdiv(c1, c2):
    return _div(hermmul, c1, c2)


def herme2poly(input):
    [input] = _as_series([input])
    n = len(input)
    if n == 1:
        return input
    if n == 2:
        return input
    else:
        c0 = input[-2]
        c1 = input[-1]

        for i in range(n - 1, 1, -1):
            tmp = c0
            c0 = polysub(input[i - 2], c1 * (i - 1))
            c1 = polyadd(tmp, polymulx(c1))
        return polyadd(c0, polymulx(c1))


def hermecompanion(c):
    [c] = _as_series([c])
    if len(c) < 2:
        raise ValueError("Series must have maximum degree of at least 1.")
    if len(c) == 2:
        return torch.tensor([[-c[0] / c[1]]])

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


def hermeder(c, m=1, scl=1, axis=0):
    c = numpy.array(c, ndmin=1)
    if c.dtype.char in "?bBhHiIlLqQpP":
        c = c.astype(numpy.double)
    cnt = operator.index(m)
    iaxis = operator.index(axis)
    if cnt < 0:
        raise ValueError("The order of derivation must be non-negative")
    iaxis = _normalize_axis_index(iaxis, c.ndim)

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


def hermediv(c1, c2):
    return _div(hermemul, c1, c2)


def hermefit(x, y, deg, rcond=None, full=False, w=None):
    return _fit(hermevander, x, y, deg, rcond, full, w)


def hermefromroots(input):
    return _from_roots(hermeline, hermemul, input)


def hermegauss(input):
    ideg = operator.index(input)
    if ideg <= 0:
        raise ValueError("deg must be a positive integer")

    c = numpy.array([0] * input + [1])
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


def hermegrid2d(x, y, c):
    return _grid(hermeval, c, x, y)


def hermegrid3d(x, y, z, c):
    return _grid(hermeval, c, x, y, z)


def hermeint(c, m=1, k=None, lbnd=0, scl=1, axis=0):
    if k is None:
        k = []
    c = numpy.array(c, ndmin=1)
    if c.dtype.char in "?bBhHiIlLqQpP":
        c = c.astype(numpy.double)
    if not numpy.iterable(k):
        k = [k]
    cnt = operator.index(m)
    iaxis = operator.index(axis)
    if cnt < 0:
        raise ValueError("The order of integration must be non-negative")
    if len(k) > cnt:
        raise ValueError("Too many integration constants")
    if numpy.ndim(lbnd) != 0:
        raise ValueError("lbnd must be a scalar.")
    if numpy.ndim(scl) != 0:
        raise ValueError("scl must be a scalar.")
    iaxis = _normalize_axis_index(iaxis, c.ndim)

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


def hermeline(input, other):
    if other != 0:
        return torch.tensor([input, other])
    else:
        return torch.tensor([input])


def hermemul(c1, c2):
    [c1, c2] = _as_series([c1, c2])

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


def hermemulx(c):
    [c] = _as_series([c])

    if len(c) == 1 and c[0] == 0:
        return c

    prd = numpy.empty(len(c) + 1, dtype=c.dtype)
    prd[0] = c[0] * 0
    prd[1] = c[0]
    for i in range(1, len(c)):
        prd[i + 1] = c[i]
        prd[i - 1] += c[i] * i
    return prd


def hermepow(c, pow, maxpower=16):
    return _pow(hermemul, c, pow, maxpower)


def hermeroots(input):
    [input] = _as_series([input])
    if len(input) <= 1:
        return numpy.array([], dtype=input.dtype)
    if len(input) == 2:
        return numpy.array([-input[0] / input[1]])

    m = hermecompanion(input)[::-1, ::-1]
    r = numpy.linalg.eigvals(m)
    r.sort()
    return r


def hermeval(x, c, tensor=True):
    c = numpy.array(c, ndmin=1)
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
    return _evaluate(hermeval, c, x, y)


def hermeval3d(x, y, z, c):
    return _evaluate(hermeval, c, x, y, z)


def hermevander(x, deg):
    ideg = operator.index(deg)
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
            v[i] = v[i - 1] * x - v[i - 2] * (i - 1)
    return numpy.moveaxis(v, 0, -1)


def hermevander2d(x, y, deg):
    return _vander_nd_flat((hermevander, hermevander), (x, y), deg)


def hermevander3d(x, y, z, deg):
    return _vander_nd_flat((hermevander, hermevander, hermevander), (x, y, z), deg)


def hermfit(x, y, deg, rcond=None, full=False, w=None):
    return _fit(hermvander, x, y, deg, rcond, full, w)


def hermfromroots(input):
    return _from_roots(hermline, hermmul, input)


def hermgauss(input):
    ideg = operator.index(input)
    if ideg <= 0:
        raise ValueError("deg must be a positive integer")

    c = numpy.array([0] * input + [1], dtype=numpy.float64)
    m = hermcompanion(c)
    output = numpy.linalg.eigvalsh(m)

    dy = _normed_hermite_n(output, ideg)
    df = _normed_hermite_n(output, ideg - 1) * numpy.sqrt(2 * ideg)
    output -= dy / df

    fm = _normed_hermite_n(output, ideg - 1)
    fm /= numpy.abs(fm).max()
    weight = 1 / (fm * fm)

    weight = (weight + weight[::-1]) / 2
    output = (output - output[::-1]) / 2

    weight = weight * (numpy.sqrt(numpy.pi) / numpy.sum(weight))

    return output, weight


def hermgrid2d(x, y, c):
    return _grid(hermval, c, x, y)


def hermgrid3d(x, y, z, c):
    return _grid(hermval, c, x, y, z)


def hermint(c, m=1, k=None, lbnd=0, scl=1, axis=0):
    if k is None:
        k = []
    c = numpy.array(c, ndmin=1)
    if c.dtype.char in "?bBhHiIlLqQpP":
        c = c.astype(numpy.double)
    if not numpy.iterable(k):
        k = [k]
    cnt = operator.index(m)
    iaxis = operator.index(axis)
    if cnt < 0:
        raise ValueError("The order of integration must be non-negative")
    if len(k) > cnt:
        raise ValueError("Too many integration constants")
    if numpy.ndim(lbnd) != 0:
        raise ValueError("lbnd must be a scalar.")
    if numpy.ndim(scl) != 0:
        raise ValueError("scl must be a scalar.")
    iaxis = _normalize_axis_index(iaxis, c.ndim)

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


def hermmul(c1, c2):
    [c1, c2] = _as_series([c1, c2])

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


def hermmulx(c):
    [c] = _as_series([c])

    if len(c) == 1 and c[0] == 0:
        return c

    prd = numpy.empty(len(c) + 1, dtype=c.dtype)
    prd[0] = c[0] * 0
    prd[1] = c[0] / 2
    for i in range(1, len(c)):
        prd[i + 1] = c[i] / 2
        prd[i - 1] += c[i] * i
    return prd


def hermpow(c, pow, maxpower=16):
    return _pow(hermmul, c, pow, maxpower)


def hermroots(input):
    [input] = _as_series([input])
    if len(input) <= 1:
        return numpy.array([], dtype=input.dtype)
    if len(input) == 2:
        return numpy.array([-0.5 * input[0] / input[1]])

    m = hermcompanion(input)[::-1, ::-1]
    r = numpy.linalg.eigvals(m)
    r.sort()
    return r


def hermval(x, c, tensor=True):
    c = numpy.array(c, ndmin=1)
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
    return _evaluate(hermval, c, x, y)


def hermval3d(x, y, z, c):
    return _evaluate(hermval, c, x, y, z)


def hermvander(x, deg):
    ideg = operator.index(deg)
    if ideg < 0:
        raise ValueError("deg must be non-negative")

    x = numpy.array(x, ndmin=1) + 0.0
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


def lag2poly(c):
    [c] = _as_series([c])
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


def lagcompanion(c):
    [c] = _as_series([c])
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


def lagder(c, m=1, scl=1, axis=0):
    c = numpy.array(c, ndmin=1)
    if c.dtype.char in "?bBhHiIlLqQpP":
        c = c.astype(numpy.double)

    cnt = operator.index(m)
    iaxis = operator.index(axis)
    if cnt < 0:
        raise ValueError("The order of derivation must be non-negative")
    iaxis = _normalize_axis_index(iaxis, c.ndim)

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


def lagdiv(c1, c2):
    return _div(lagmul, c1, c2)


def lagfit(x, y, deg, rcond=None, full=False, w=None):
    return _fit(lagvander, x, y, deg, rcond, full, w)


def lagfromroots(roots):
    return _from_roots(lagline, lagmul, roots)


def laggauss(input):
    ideg = operator.index(input)
    if ideg <= 0:
        raise ValueError("deg must be a positive integer")

    c = numpy.array([0] * input + [1])
    m = lagcompanion(c)
    x = numpy.linalg.eigvalsh(m)

    dy = lagval(x, c)
    df = lagval(x, lagder(c))
    x -= dy / df

    fm = lagval(x, c[1:])

    fm = fm / numpy.max(numpy.abs(fm))
    df = df / numpy.max(numpy.abs(df))

    weight = 1.0 / (fm * df)

    weight = weight / numpy.sum(weight)

    return x, weight


def laggrid2d(x, y, c):
    return _grid(lagval, c, x, y)


def laggrid3d(x, y, z, c):
    return _grid(lagval, c, x, y, z)


def lagint(c, m=1, k=None, lbnd=0, scl=1, axis=0):
    if k is None:
        k = []
    c = numpy.array(c, ndmin=1)
    if c.dtype.char in "?bBhHiIlLqQpP":
        c = c.astype(numpy.double)
    if not numpy.iterable(k):
        k = [k]
    cnt = operator.index(m)
    iaxis = operator.index(axis)
    if cnt < 0:
        raise ValueError("The order of integration must be non-negative")
    if len(k) > cnt:
        raise ValueError("Too many integration constants")
    if numpy.ndim(lbnd) != 0:
        raise ValueError("lbnd must be a scalar.")
    if numpy.ndim(scl) != 0:
        raise ValueError("scl must be a scalar.")
    iaxis = _normalize_axis_index(iaxis, c.ndim)

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


def lagmul(c1, c2):
    [c1, c2] = _as_series([c1, c2])

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


def lagmulx(c):
    [c] = _as_series([c])

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


def lagpow(c, pow, maxpower=16):
    return _pow(lagmul, c, pow, maxpower)


def lagroots(c):
    [c] = _as_series([c])
    if len(c) <= 1:
        return numpy.array([], dtype=c.dtype)
    if len(c) == 2:
        return numpy.array([1 + c[0] / c[1]])

    m = lagcompanion(c)[::-1, ::-1]
    r = numpy.linalg.eigvals(m)
    r.sort()
    return r


def lagval(x, c, tensor=True):
    c = numpy.array(c, ndmin=1)
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
    return _evaluate(lagval, c, x, y)


def lagval3d(x, y, z, c):
    return _evaluate(lagval, c, x, y, z)


def lagvander(x, deg):
    ideg = operator.index(deg)
    if ideg < 0:
        raise ValueError("deg must be non-negative")

    x = numpy.array(x, ndmin=1) + 0.0
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


def leg2poly(c):
    [c] = _as_series([c])
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


def legcompanion(c):
    [c] = _as_series([c])
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


def legder(c, m=1, scl=1, axis=0):
    c = numpy.array(c, ndmin=1)
    if c.dtype.char in "?bBhHiIlLqQpP":
        c = c.astype(numpy.double)
    cnt = operator.index(m)
    iaxis = operator.index(axis)
    if cnt < 0:
        raise ValueError("The order of derivation must be non-negative")
    iaxis = _normalize_axis_index(iaxis, c.ndim)

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


def legdiv(c1, c2):
    return _div(legmul, c1, c2)


def legfit(x, y, deg, rcond=None, full=False, w=None):
    return _fit(legvander, x, y, deg, rcond, full, w)


def legfromroots(roots):
    return _from_roots(legline, legmul, roots)


def leggauss(input):
    ideg = operator.index(input)
    if ideg <= 0:
        raise ValueError("deg must be a positive integer")

    c = numpy.array([0] * input + [1])
    m = legcompanion(c)
    x = numpy.linalg.eigvalsh(m)

    dy = legval(x, c)
    df = legval(x, legder(c))
    x -= dy / df

    fm = legval(x, c[1:])
    fm /= numpy.abs(fm).max()
    df /= numpy.abs(df).max()
    weights = 1 / (fm * df)

    weights = (weights + weights[::-1]) / 2
    x = (x - x[::-1]) / 2

    weights = weights * (2.0 / weights.sum())

    return x, weights


def leggrid2d(x, y, c):
    return _grid(legval, c, x, y)


def leggrid3d(x, y, z, c):
    return _grid(legval, c, x, y, z)


def legint(c, m=1, k=None, lbnd=0, scl=1, axis=0):
    if k is None:
        k = []
    c = numpy.array(c, ndmin=1)
    if c.dtype.char in "?bBhHiIlLqQpP":
        c = c.astype(numpy.double)
    if not numpy.iterable(k):
        k = [k]
    cnt = operator.index(m)
    iaxis = operator.index(axis)
    if cnt < 0:
        raise ValueError("The order of integration must be non-negative")
    if len(k) > cnt:
        raise ValueError("Too many integration constants")
    if numpy.ndim(lbnd) != 0:
        raise ValueError("lbnd must be a scalar.")
    if numpy.ndim(scl) != 0:
        raise ValueError("scl must be a scalar.")
    iaxis = _normalize_axis_index(iaxis, c.ndim)

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


def legmul(c1, c2):
    [c1, c2] = _as_series([c1, c2])

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


def legmulx(c):
    [c] = _as_series([c])

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


def legpow(c, pow, maxpower=16):
    return _pow(legmul, c, pow, maxpower)


def legroots(c):
    [c] = _as_series([c])
    if len(c) < 2:
        return numpy.array([], dtype=c.dtype)
    if len(c) == 2:
        return numpy.array([-c[0] / c[1]])

    m = legcompanion(c)[::-1, ::-1]
    r = numpy.linalg.eigvals(m)
    r.sort()
    return r


def legval(x, c, tensor=True):
    c = numpy.array(c, ndmin=1)
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
    return _evaluate(legval, c, x, y)


def legval3d(x, y, z, c):
    return _evaluate(legval, c, x, y, z)


def legvander(x, deg):
    ideg = operator.index(deg)
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
            v[i] = (v[i - 1] * x * (2 * i - 1) - v[i - 2] * (i - 1)) / i
    return numpy.moveaxis(v, 0, -1)


def legvander2d(x, y, deg):
    return _vander_nd_flat((legvander, legvander), (x, y), deg)


def legvander3d(x, y, z, deg):
    return _vander_nd_flat((legvander, legvander, legvander), (x, y, z), deg)


def poly2cheb(input):
    [input] = _as_series([input])

    output = 0

    for index in range(len(input) - 1, -1, -1):
        output = chebmulx(output)

        output = _add(output, input[index])

    return output


def poly2herm(input):
    [input] = _as_series([input])
    deg = len(input) - 1
    res = 0
    for i in range(deg, -1, -1):
        res = hermadd(hermmulx(res), input[i])
    return res


def poly2herme(input):
    [input] = _as_series([input])
    deg = len(input) - 1
    res = 0
    for i in range(deg, -1, -1):
        res = hermeadd(hermemulx(res), input[i])
    return res


def poly2lag(pol):
    [pol] = _as_series([pol])
    res = 0
    for p in pol[::-1]:
        res = lagadd(lagmulx(res), p)
    return res


def poly2leg(pol):
    [pol] = _as_series([pol])
    deg = len(pol) - 1
    res = 0
    for i in range(deg, -1, -1):
        res = legadd(legmulx(res), pol[i])
    return res


def polycompanion(series):
    (series,) = _as_series([series])

    if len(series) < 2:
        raise ValueError

    if len(series) == 2:
        output = numpy.array([[-series[0] / series[1]]])
    else:
        n = series.shape[-1] - 1

        output = numpy.zeros([n, n], dtype=series.dtype)

        bot = numpy.reshape(output, -1)[n :: n + 1]

        bot[...] = 1

        output[:, -1] = output[:, -1] - (series[:-1] / series[-1])

    return output


def polyder(input, order: int = 1, scale: float = 1, axis: int = 0):
    output = numpy.array(input, ndmin=1)

    if output.dtype.char in "?bBhHiIlLqQpP":
        output = output + 0.0

    dtype = output.dtype

    cnt = operator.index(order)

    axis = operator.index(axis)

    if cnt < 0:
        raise ValueError

    axis = _normalize_axis_index(axis, output.ndim)

    if cnt == 0:
        return output

    output = numpy.moveaxis(output, axis, 0)

    n = len(output)

    if cnt >= n:
        output = output[:1] * 0
    else:
        for _ in range(cnt):
            n = n - 1

            output = output * scale

            der = numpy.empty((n,) + output.shape[1:], dtype=dtype)

            for j in range(n, 0, -1):
                der[j - 1] = j * output[j]

            output = der

    output = numpy.moveaxis(output, 0, axis)

    return output


def polydiv(c1, c2):
    [c1, c2] = _as_series([c1, c2])
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
        return c1[j + 1 :] / scl, _trim_sequence(c1[: j + 1])


def polyfit(x, y, deg, rcond=None, full=False, w=None):
    return _fit(polyvander, x, y, deg, rcond, full, w)


def polyfromroots(roots):
    return _from_roots(polyline, polymul, roots)


def polygrid2d(x, y, c):
    return _grid(polyval, c, x, y)


def polygrid3d(x, y, z, c):
    return _grid(polyval, c, x, y, z)


def polyint(c, m=1, k=None, lbnd=0, scl=1, axis=0):
    if k is None:
        k = []
    c = numpy.array(c, ndmin=1)
    if c.dtype.char in "?bBhHiIlLqQpP":
        c = c + 0.0
    cdt = c.dtype
    if not numpy.iterable(k):
        k = [k]
    cnt = operator.index(m)
    iaxis = operator.index(axis)
    if cnt < 0:
        raise ValueError("The order of integration must be non-negative")
    if len(k) > cnt:
        raise ValueError("Too many integration constants")
    if numpy.ndim(lbnd) != 0:
        raise ValueError("lbnd must be a scalar.")
    if numpy.ndim(scl) != 0:
        raise ValueError("scl must be a scalar.")
    iaxis = _normalize_axis_index(iaxis, c.ndim)

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


def polymul(input, other):
    input, other = _as_series([input, other])

    output = numpy.convolve(input, other)

    output = _trim_sequence(output)

    return output


def polymulx(input):
    (input,) = _as_series([input])

    if len(input) == 1 and input[0] == 0:
        return input

    output = numpy.empty(len(input) + 1, dtype=input.dtype)

    output[0] = input[0] * 0.0

    output[1:] = input

    return output


def polypow(c, pow, maxpower=None):
    return _pow(numpy.convolve, c, pow, maxpower)


def polyroots(series):
    (series,) = _as_series([series])

    if len(series) < 2:
        return numpy.array([], dtype=series.dtype)

    if len(series) == 2:
        return numpy.array([-series[0] / series[1]])

    output = polycompanion(series)

    output = numpy.flip(output, axis=0)
    output = numpy.flip(output, axis=1)

    output = numpy.linalg.eigvals(output)

    output = numpy.sort(output)

    return output


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


def polyval2d(x, y, c):
    return _evaluate(polyval, c, x, y)


def polyval3d(x, y, z, c):
    return _evaluate(polyval, c, x, y, z)


def polyvalfromroots(x, output, tensor=True):
    output = numpy.array(output, ndmin=1)

    if output.dtype.char in "?bBhHiIlLqQpP":
        output = output.astype(numpy.float64)

    if isinstance(x, (tuple, list)):
        x = numpy.asarray(x)

    if isinstance(x, numpy.ndarray):
        if tensor:
            shape = (1,) * x.ndim

            output = numpy.reshape(output, [*output.shape, *shape])
        elif x.ndim >= output.ndim:
            raise ValueError

    return numpy.prod(x - output, axis=0)


def polyvander(input, degree):
    degree = operator.index(degree)

    if degree < 0:
        raise ValueError

    input = numpy.array(input, ndmin=1) + 0.0

    output = numpy.empty([degree + 1, *input.shape], dtype=input.dtype)

    output[0] = input * 0.0 + 1.0

    if degree > 0:
        output[1] = input

        for i in range(2, degree + 1):
            output[i] = output[i - 1] * input

    output = numpy.moveaxis(output, 0, -1)

    return output


def polyvander2d(x, y, deg):
    return _vander_nd_flat((polyvander, polyvander), (x, y), deg)


def polyvander3d(x, y, z, deg):
    return _vander_nd_flat((polyvander, polyvander, polyvander), (x, y, z), deg)


__all__ = [
    "cheb2poly",
    "chebcompanion",
    "chebder",
    "chebdiv",
    "chebdomain",
    "chebfit",
    "chebfromroots",
    "chebgauss",
    "chebgrid2d",
    "chebgrid3d",
    "chebint",
    "chebinterpolate",
    "chebline",
    "chebmul",
    "chebmulx",
    "chebone",
    "chebpow",
    "chebpts1",
    "chebpts2",
    "chebroots",
    "chebtrim",
    "chebval",
    "chebval2d",
    "chebval3d",
    "chebvander",
    "chebvander2d",
    "chebvander3d",
    "chebweight",
    "chebx",
    "chebzero",
    "herm2poly",
    "hermcompanion",
    "hermder",
    "hermdiv",
    "herme2poly",
    "hermecompanion",
    "hermeder",
    "hermediv",
    "hermefit",
    "hermefromroots",
    "hermegauss",
    "hermegrid2d",
    "hermegrid3d",
    "hermeint",
    "hermeline",
    "hermemul",
    "hermemulx",
    "hermepow",
    "hermeroots",
    "hermetrim",
    "hermeval",
    "hermeval2d",
    "hermeval3d",
    "hermevander",
    "hermevander2d",
    "hermevander3d",
    "hermfit",
    "hermfromroots",
    "hermgauss",
    "hermgrid2d",
    "hermgrid3d",
    "hermint",
    "hermmul",
    "hermmulx",
    "hermpow",
    "hermroots",
    "hermtrim",
    "hermval",
    "hermval2d",
    "hermval3d",
    "hermvander",
    "hermvander2d",
    "hermvander3d",
    "lag2poly",
    "lagcompanion",
    "lagder",
    "lagdiv",
    "lagfit",
    "lagfromroots",
    "laggauss",
    "laggrid2d",
    "laggrid3d",
    "lagint",
    "lagmul",
    "lagmulx",
    "lagpow",
    "lagroots",
    "lagtrim",
    "lagval",
    "lagval2d",
    "lagval3d",
    "lagvander",
    "lagvander2d",
    "lagvander3d",
    "leg2poly",
    "legcompanion",
    "legder",
    "legdiv",
    "legfit",
    "legfromroots",
    "leggauss",
    "leggrid2d",
    "leggrid3d",
    "legint",
    "legmul",
    "legmulx",
    "legpow",
    "legroots",
    "legtrim",
    "legval",
    "legval2d",
    "legval3d",
    "legvander",
    "legvander2d",
    "legvander3d",
    "poly2cheb",
    "poly2herm",
    "poly2herme",
    "poly2lag",
    "poly2leg",
    "polycompanion",
    "polyder",
    "polydiv",
    "polyfit",
    "polyfromroots",
    "polygrid2d",
    "polygrid3d",
    "polyint",
    "polymul",
    "polymulx",
    "polypow",
    "polyroots",
    "polytrim",
    "polyval",
    "polyval2d",
    "polyval3d",
    "polyvalfromroots",
    "polyvander",
    "polyvander2d",
    "polyvander3d",
]
