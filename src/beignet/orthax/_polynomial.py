import functools
import operator

import jax
import jax.numpy
import numpy

chebdomain = jax.numpy.array([-1, 1])
chebone = jax.numpy.array([1])
chebx = jax.numpy.array([0, 1])
chebzero = jax.numpy.array([0])
hermdomain = jax.numpy.array([-1, 1])
hermedomain = jax.numpy.array([-1, 1])
hermeone = jax.numpy.array([1])
hermex = jax.numpy.array([0, 1])
hermezero = jax.numpy.array([0])
hermone = jax.numpy.array([1])
hermx = jax.numpy.array([0, 1 / 2])
hermzero = jax.numpy.array([0])
lagdomain = jax.numpy.array([0, 1])
lagone = jax.numpy.array([1])
lagx = jax.numpy.array([1, -1])
lagzero = jax.numpy.array([0])
legdomain = jax.numpy.array([-1, 1])
legone = jax.numpy.array([1])
legx = jax.numpy.array([0, 1])
legzero = jax.numpy.array([0])
polydomain = jax.numpy.array([-1, 1])
polyone = jax.numpy.array([1])
polyx = jax.numpy.array([0, 1])
polyzero = jax.numpy.array([0])


def _nth_slice(i, ndim):
    sl = [jax.numpy.newaxis] * ndim
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

    points = tuple(jax.numpy.array(tuple(points), copy=False) + 0.0)

    vander_arrays = (
        vander_fs[i](points[i], degrees[i])[(...,) + _nth_slice(i, n_dims)]
        for i in range(n_dims)
    )

    return functools.reduce(operator.mul, vander_arrays)


def _vander_nd_flat(vander_fs, points, degrees):
    v = _vander_nd(vander_fs, points, degrees)
    return v.reshape(v.shape[: -len(degrees)] + (-1,))


def _fromroots(line_f, mul_f, roots):
    roots = jax.numpy.asarray(roots)
    if roots.size == 0:
        return jax.numpy.ones(1)

    roots = jax.numpy.sort(roots)

    retlen = len(roots) + 1

    def p_scan_fun(carry, x):
        return carry, _add(jax.numpy.zeros(retlen, dtype=x.dtype), line_f(-x, 1))

    _, p = jax.lax.scan(p_scan_fun, 0, roots)

    p = jax.numpy.asarray(p)
    n = len(p)

    def cond_fun(val):
        return val[0] > 1

    def body_fun(val):
        m, r = divmod(val[0], 2)
        arr = val[1]
        tmp = jax.numpy.array([jax.numpy.zeros(retlen, dtype=p.dtype)] * len(p))

        def inner_body_fun(i, val):
            return val.at[i].set(mul_f(arr[i], arr[i + m])[:retlen])

        tmp = jax.lax.fori_loop(0, m, inner_body_fun, tmp)
        tmp = jax.lax.cond(
            r, lambda x: x.at[0].set(mul_f(x[0], arr[2 * m])[:retlen]), lambda x: x, tmp
        )

        return (m, tmp)

    _, ret = jax.lax.while_loop(cond_fun, body_fun, (n, p))
    return ret[0]


def _valnd(val_f, c, *args):
    args = [jax.numpy.asarray(a) for a in args]
    shape0 = args[0].shape
    if not all(a.shape == shape0 for a in args[1:]):
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
    c1, c2 = as_series(c1, c2)
    lc1 = len(c1)
    lc2 = len(c2)
    if lc1 < lc2:
        return jax.numpy.zeros_like(c1[:1]), c1
    elif lc2 == 1:
        return c1 / c2[-1], jax.numpy.zeros_like(c1[:1])
    else:

        def _ldordidx(x):  # index of highest order nonzero term
            return len(x) - 1 - jax.numpy.nonzero(x[::-1], size=1)[0][0]

        quo = jax.numpy.zeros(lc1 - lc2 + 1, dtype=c1.dtype)
        rem = c1
        ridx = len(rem) - 1
        sz = lc1 - _ldordidx(c2) - 1
        y = jax.numpy.zeros(lc1 + lc2 + 1, dtype=c1.dtype).at[sz].set(1.0)

        def body(k, val):
            quo, rem, y, ridx = val
            i = sz - k
            p = mul_f(y, c2)
            pidx = _ldordidx(p)
            t = rem[ridx] / p[pidx]
            rem = _sub(rem.at[ridx].set(0), t * p.at[pidx].set(0))[: len(rem)]
            quo = quo.at[i].set(t)
            ridx -= 1
            y = jax.numpy.roll(y, -1)
            return quo, rem, y, ridx

        quo, rem, _, _ = jax.lax.fori_loop(0, sz, body, (quo, rem, y, ridx))
        return quo, rem


def _add(c1, c2):
    c1, c2 = as_series(c1, c2)
    if len(c1) > len(c2):
        ret = c1.at[: c2.size].add(c2)
    else:
        ret = c2.at[: c1.size].add(c1)
    return ret


def _sub(c1, c2):
    c1, c2 = as_series(c1, c2)
    if len(c1) > len(c2):
        ret = c1.at[: c2.size].add(-c2)
    else:
        ret = (-c2).at[: c1.size].add(c1)
    return ret


def _fit(vander_f, x, y, deg, rcond=None, full=False, w=None):  # noqa:C901
    x = jax.numpy.asarray(x)
    y = jax.numpy.asarray(y)
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
        lmax = int(deg)
        van = vander_f(x, lmax)
    else:
        deg = numpy.sort(deg)
        lmax = int(deg[-1])
        van = vander_f(x, lmax)[:, deg]

    lhs = van.T
    rhs = y.T
    if w is not None:
        w = jax.numpy.asarray(w)
        if w.ndim != 1:
            raise TypeError("expected 1D vector for w")
        if len(x) != len(w):
            raise TypeError("expected x and w to have same length")

        lhs = lhs * w
        rhs = rhs * w

    if rcond is None:
        rcond = len(x) * jax.numpy.finfo(x.dtype).eps

    if issubclass(lhs.dtype.type, jax.numpy.complexfloating):
        scl = jax.numpy.sqrt(
            (jax.numpy.square(lhs.real) + jax.numpy.square(lhs.imag)).sum(1)
        )
    else:
        scl = jax.numpy.sqrt(jax.numpy.square(lhs).sum(1))
    scl = jax.numpy.where(scl == 0, 1, scl)

    c, resids, rank, s = jax.numpy.linalg.lstsq(lhs.T / scl, rhs.T, rcond)
    c = (c.T / scl).T

    if deg.ndim > 0:
        if c.ndim == 2:
            cc = jax.numpy.zeros((lmax + 1, c.shape[1]), dtype=c.dtype)
        else:
            cc = jax.numpy.zeros(lmax + 1, dtype=c.dtype)
        cc = cc.at[deg].set(c)
        c = cc

    if full:
        return c, [resids, rank, s, rcond]
    else:
        return c


def _pow(mul_f, c, pow, maxpower):
    c = as_series(c)
    power = int(pow)
    if power != pow or power < 0:
        raise ValueError("Power must be a non-negative integer.")
    elif maxpower is not None and power > maxpower:
        raise ValueError("Power is too large")
    elif power == 0:
        return jax.numpy.array([1], dtype=c.dtype)
    elif power == 1:
        return c
    else:
        prd = jax.numpy.zeros(len(c) * pow, dtype=c.dtype)
        prd = _add(prd, c)

        def body(i, p):
            p = mul_f(p, c, mode="same")
            return p

        prd = jax.lax.fori_loop(2, power + 1, body, prd)
        return prd


def _pad_along_axis(array, pad=(0, 0), axis=0):
    array = jax.numpy.moveaxis(array, axis, 0)

    if pad[0] < 0:
        array = array[abs(pad[0]) :]
        pad = (0, pad[1])
    if pad[1] < 0:
        array = array[: -abs(pad[1])]
        pad = (pad[0], 0)

    npad = [(0, 0)] * array.ndim
    npad[0] = pad

    array = jax.numpy.pad(array, pad_width=npad, mode="constant", constant_values=0)
    return jax.numpy.moveaxis(array, 0, axis)


def _cseries_to_zseries(c):
    n = c.size
    zs = jax.numpy.zeros(2 * n - 1, dtype=c.dtype)
    zs = zs.at[n - 1 :].set(c / 2)
    return zs + zs[::-1]


def _zseries_to_cseries(zs):
    n = (zs.size + 1) // 2
    c = zs[n - 1 :].copy()
    c = c.at[1:n].multiply(2)
    return c


def _zseries_mul(z1, z2, mode="full"):
    return jax.numpy.convolve(z1, z2, mode=mode)


def poly2cheb(pol):
    pol = as_series(pol)
    deg = len(pol) - 1

    res = jax.numpy.zeros_like(pol)

    def body(i, res):
        k = deg - i
        res = chebadd(chebmulx(res, mode="same"), pol[k])
        return res

    res = jax.lax.fori_loop(0, deg + 1, body, res)
    return res


def cheb2poly(c):
    from .polynomial import polyadd, polymulx, polysub

    c = as_series(c)
    n = len(c)
    if n < 3:
        return c
    else:
        c0 = jax.numpy.zeros_like(c).at[0].set(c[-2])
        c1 = jax.numpy.zeros_like(c).at[0].set(c[-1])

        def body(k, c0c1):
            i = n - 1 - k
            c0, c1 = c0c1
            tmp = c0
            c0 = polysub(c[i - 2], c1)
            c1 = polyadd(tmp, polymulx(c1, "same") * 2)
            return c0, c1

        c0, c1 = jax.lax.fori_loop(0, n - 2, body, (c0, c1))

        return polyadd(c0, polymulx(c1, "same"))


def chebline(off, scl):
    return jax.numpy.array([off, scl])


def chebfromroots(roots):
    return _fromroots(chebline, chebmul, roots)


def chebadd(c1, c2):
    return _add(c1, c2)


def chebsub(c1, c2):
    return _sub(c1, c2)


def chebmulx(c, mode="full"):
    c = as_series(c)
    prd = jax.numpy.zeros(len(c) + 1, dtype=c.dtype)
    prd = prd.at[1].set(c[0])

    if len(c) > 1:
        tmp = c[1:] / 2
        prd = prd.at[2:].set(tmp)
        prd = prd.at[0:-2].add(tmp)

    if mode == "same":
        prd = prd[: len(c)]
    return prd


def chebmul(c1, c2, mode="full"):
    c1, c2 = as_series(c1, c2)
    z1 = _cseries_to_zseries(c1)
    z2 = _cseries_to_zseries(c2)
    prd = _zseries_mul(z1, z2, mode=mode)
    ret = _zseries_to_cseries(prd)
    if mode == "same":
        ret = ret[: max(len(c1), len(c2))]

    return ret


def chebdiv(c1, c2):
    return _div(chebmul, c1, c2)


def chebpow(c, pow, maxpower=16):
    c = as_series(c)
    power = int(pow)
    if power != pow or power < 0:
        raise ValueError("Power must be a non-negative integer.")
    elif maxpower is not None and power > maxpower:
        raise ValueError("Power is too large")
    elif power == 0:
        return jax.numpy.array([1], dtype=c.dtype)
    elif power == 1:
        return c
    else:
        prd = jax.numpy.zeros(len(c) * pow, dtype=c.dtype)
        prd = chebadd(prd, c)
        zs = _cseries_to_zseries(c)
        prd = _cseries_to_zseries(prd)

        def body(i, p):
            p = jax.numpy.convolve(p, zs, mode="same")
            return p

        prd = jax.lax.fori_loop(2, power + 1, body, prd)
        return _zseries_to_cseries(prd)


def chebder(c, m=1, scl=1, axis=0):
    if m < 0:
        raise ValueError("The order of derivation must be non-negative")

    c = as_series(c)

    if m == 0:
        return c

    c = jax.numpy.moveaxis(c, axis, 0)
    n = len(c)
    if m >= n:
        c = jax.numpy.zeros_like(c[:1])
    else:
        for _ in range(m):
            n = n - 1
            c *= scl
            der = jax.numpy.empty((n,) + c.shape[1:], dtype=c.dtype)

            def body(k, der_c, n=n):
                j = n - k
                der, c = der_c
                der = der.at[j - 1].set((2 * j) * c[j])
                c = c.at[j - 2].add((j * c[j]) / (j - 2))
                return der, c

            der, c = jax.lax.fori_loop(0, n - 2, body, (der, c))
            if n > 1:
                der = der.at[1].set(4 * c[2])
            der = der.at[0].set(c[1])
            c = der

    c = jax.numpy.moveaxis(c, 0, axis)
    return c


def chebint(c, m=1, k=None, lbnd=0, scl=1, axis=0):
    if k is None:
        k = []
    c = as_series(c)
    lbnd, scl = map(jax.numpy.asarray, (lbnd, scl))

    if not jax.numpy.iterable(k):
        k = [k]
    if len(k) > m:
        raise ValueError("Too many integration constants")
    if jax.numpy.ndim(lbnd) != 0:
        raise ValueError("lbnd must be a scalar.")
    if jax.numpy.ndim(scl) != 0:
        raise ValueError("scl must be a scalar.")

    if m == 0:
        return c

    c = jax.numpy.moveaxis(c, axis, 0)
    k = jax.numpy.array(list(k) + [0] * (m - len(k)), ndmin=1)

    for i in range(m):
        n = len(c)
        c *= scl
        tmp = jax.numpy.empty((n + 1,) + c.shape[1:], dtype=c.dtype)
        tmp = tmp.at[0].set(c[0] * 0)
        tmp = tmp.at[1].set(c[0])
        if n > 1:
            tmp = tmp.at[2].set(c[1] / 4)
        j = jax.numpy.arange(2, n)
        tmp = tmp.at[j + 1].set((c[j].T / (2 * (j + 1))).T)
        tmp = tmp.at[j - 1].add(-(c[j].T / (2 * (j - 1))).T)
        tmp = tmp.at[0].add(k[i] - chebval(lbnd, tmp))
        c = tmp
    c = jax.numpy.moveaxis(c, 0, axis)
    return c


def chebval(x, c, tensor=True):
    c = as_series(c)
    x = jax.numpy.asarray(x)
    if tensor:
        c = c.reshape(c.shape + (1,) * x.ndim)

    if len(c) == 1:
        c0 = c[0]
        c1 = 0
    elif len(c) == 2:
        c0 = c[0]
        c1 = c[1]
    else:
        x2 = 2 * x
        c0 = c[-2] * jax.numpy.ones_like(x)
        c1 = c[-1] * jax.numpy.ones_like(x)

        def body(i, val):
            c0, c1 = val
            tmp = c0
            c0 = c[-i] - c1
            c1 = tmp + c1 * x2
            return c0, c1

        c0, c1 = jax.lax.fori_loop(3, len(c) + 1, body, (c0, c1))

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
    if deg < 0:
        raise ValueError("deg must be non-negative")

    x = jax.numpy.array(x, ndmin=1)
    dims = (deg + 1,) + x.shape
    dtyp = jax.numpy.promote_types(x.dtype, jax.numpy.array(0.0).dtype)
    x = x.astype(dtyp)
    v = jax.numpy.empty(dims, dtype=dtyp)
    v = v.at[0].set(jax.numpy.ones_like(x))

    if deg > 0:
        v = v.at[1].set(x)
        x2 = 2 * x

        def body(i, v):
            return v.at[i].set(v[i - 1] * x2 - v[i - 2])

        v = jax.lax.fori_loop(2, deg + 1, body, v)

    return jax.numpy.moveaxis(v, 0, -1)


def chebvander2d(x, y, deg):
    return _vander_nd_flat((chebvander, chebvander), (x, y), deg)


def chebvander3d(x, y, z, deg):
    return _vander_nd_flat((chebvander, chebvander, chebvander), (x, y, z), deg)


def chebfit(x, y, deg, rcond=None, full=False, w=None):
    return _fit(chebvander, x, y, deg, rcond, full, w)


def chebcompanion(c):
    c = as_series(c)
    if len(c) < 2:
        raise ValueError("Series must have maximum degree of at least 1.")
    if len(c) == 2:
        return jax.numpy.array([[-c[0] / c[1]]])

    n = len(c) - 1
    mat = jax.numpy.zeros((n, n), dtype=c.dtype)
    scl = jax.numpy.ones(n).at[1:].set(jax.numpy.sqrt(0.5))
    shp = mat.shape
    mat = mat.flatten()
    mat = mat.at[1 :: n + 1].set(
        jax.numpy.full(n - 1, 1 / 2).at[0].set(jax.numpy.sqrt(0.5))
    )
    mat = mat.at[n :: n + 1].set(
        jax.numpy.full(n - 1, 1 / 2).at[0].set(jax.numpy.sqrt(0.5))
    )
    mat = mat.reshape(shp)
    mat = mat.at[:, -1].add(-(c[:-1] / c[-1]) * (scl / scl[-1]) * 0.5)
    return mat


def chebroots(c):
    c = as_series(c)
    if len(c) <= 1:
        return jax.numpy.array([], dtype=c.dtype)
    if len(c) == 2:
        return jax.numpy.array([-c[0] / c[1]])

    m = chebcompanion(c)[::-1, ::-1]
    r = jax.numpy.linalg.eigvals(m)
    r = jax.numpy.sort(r)
    return r


def chebinterpolate(func, deg, args=()):
    _deg = int(deg)
    if _deg != deg:
        raise ValueError("deg must be integer")
    if _deg < 0:
        raise ValueError("expected deg >= 0")

    order = _deg + 1
    xcheb = chebpts1(order)
    yfunc = func(xcheb, *args)
    m = chebvander(xcheb, _deg)
    c = jax.numpy.dot(m.T, yfunc)
    c = c.at[0].divide(order)
    c = c.at[1:].divide(0.5 * order)

    return c


def chebgauss(deg):
    deg = int(deg)
    if deg <= 0:
        raise ValueError("deg must be a positive integer")

    x = jax.numpy.cos(jax.numpy.pi * jax.numpy.arange(1, 2 * deg, 2) / (2.0 * deg))
    w = jax.numpy.ones(deg) * (jax.numpy.pi / deg)

    return x, w


def chebweight(x):
    x = jax.numpy.asarray(x)
    w = 1.0 / (jax.numpy.sqrt(1.0 + x) * jax.numpy.sqrt(1.0 - x))
    return w


def chebpts1(npts):
    _npts = int(npts)
    if _npts != npts:
        raise ValueError("npts must be integer")
    if _npts < 1:
        raise ValueError("npts must be >= 1")

    x = 0.5 * jax.numpy.pi / _npts * jax.numpy.arange(-_npts + 1, _npts + 1, 2)
    return jax.numpy.sin(x)


def chebpts2(npts):
    _npts = int(npts)
    if _npts != npts:
        raise ValueError("npts must be integer")
    if _npts < 2:
        raise ValueError("npts must be >= 2")

    x = jax.numpy.linspace(-jax.numpy.pi, 0, _npts)
    return jax.numpy.cos(x)


def poly2herm(pol):
    pol = as_series(pol)
    deg = len(pol) - 1
    res = jax.numpy.zeros_like(pol)

    def body(i, res):
        k = deg - i
        res = hermadd(hermmulx(res, mode="same"), pol[k])
        return res

    res = jax.lax.fori_loop(0, deg + 1, body, res)
    return res


def herm2poly(c):
    from .polynomial import polyadd, polymulx, polysub

    c = as_series(c)
    n = len(c)
    if n == 1:
        return c
    if n == 2:
        c = c.at[1].multiply(2)
        return c
    else:
        c0 = jax.numpy.zeros_like(c).at[0].set(c[-2])
        c1 = jax.numpy.zeros_like(c).at[0].set(c[-1])

        def body(k, c0c1):
            i = n - 1 - k
            c0, c1 = c0c1
            tmp = c0
            c0 = polysub(c[i - 2], c1 * (2 * (i - 1)))
            c1 = polyadd(tmp, polymulx(c1, "same") * 2)
            return c0, c1

        c0, c1 = jax.lax.fori_loop(0, n - 2, body, (c0, c1))

        return polyadd(c0, polymulx(c1, "same") * 2)


def hermline(off, scl):
    return jax.numpy.array([off, scl / 2])


def hermfromroots(roots):
    return _fromroots(hermline, hermmul, roots)


def hermadd(c1, c2):
    return _add(c1, c2)


def hermsub(c1, c2):
    return _sub(c1, c2)


def hermmulx(c, mode="full"):
    c = as_series(c)
    prd = jax.numpy.zeros(len(c) + 1, dtype=c.dtype)
    prd = prd.at[1].set(c[0] / 2)

    i = jax.numpy.arange(1, len(c))

    prd = prd.at[i + 1].set(c[i] / 2)
    prd = prd.at[i - 1].add(c[i] * i)

    if mode == "same":
        prd = prd[: len(c)]
    return prd


def hermmul(c1, c2, mode="full"):
    c1, c2 = as_series(c1, c2)
    lc1, lc2 = len(c1), len(c2)
    if lc1 > lc2:
        c = c2
        xs = c1
    else:
        c = c1
        xs = c2

    if len(c) == 1:
        c0 = hermadd(jax.numpy.zeros(lc1 + lc2 - 1), c[0] * xs)
        c1 = jax.numpy.zeros(lc1 + lc2 - 1)
    elif len(c) == 2:
        c0 = hermadd(jax.numpy.zeros(lc1 + lc2 - 1), c[0] * xs)
        c1 = hermadd(jax.numpy.zeros(lc1 + lc2 - 1), c[1] * xs)
    else:
        nd = len(c)
        c0 = hermadd(jax.numpy.zeros(lc1 + lc2 - 1), c[-2] * xs)
        c1 = hermadd(jax.numpy.zeros(lc1 + lc2 - 1), c[-1] * xs)

        def body(i, val):
            c0, c1, nd = val
            tmp = c0
            nd = nd - 1
            c0 = hermsub(c[-i] * xs, c1 * (2 * (nd - 1)))
            c1 = hermadd(tmp, hermmulx(c1, "same") * 2)
            return c0, c1, nd

        c0, c1, _ = jax.lax.fori_loop(3, len(c) + 1, body, (c0, c1, nd))

    ret = hermadd(c0, hermmulx(c1, "same") * 2)
    if mode == "same":
        ret = ret[: max(lc1, lc2)]
    return ret


def hermdiv(c1, c2):
    return _div(hermmul, c1, c2)


def hermpow(c, pow, maxpower=16):
    return _pow(hermmul, c, pow, maxpower)


def hermder(c, m=1, scl=1, axis=0):
    if m < 0:
        raise ValueError("The order of derivation must be non-negative")

    c = as_series(c)

    if m == 0:
        return c

    c = jax.numpy.moveaxis(c, axis, 0)
    n = len(c)
    if m >= n:
        c = jax.numpy.zeros_like(c[:1])
    else:
        for _ in range(m):
            n = n - 1
            c *= scl
            der = jax.numpy.empty((n,) + c.shape[1:], dtype=c.dtype)
            j = jax.numpy.arange(n, 0, -1)
            der = der.at[j - 1].set((2 * j * c[j].T).T)
            c = der
    c = jax.numpy.moveaxis(c, 0, axis)
    return c


def hermint(c, m=1, k=None, lbnd=0, scl=1, axis=0):
    if k is None:
        k = []
    c = as_series(c)
    lbnd, scl = map(jax.numpy.asarray, (lbnd, scl))

    if not jax.numpy.iterable(k):
        k = [k]
    if len(k) > m:
        raise ValueError("Too many integration constants")
    if jax.numpy.ndim(lbnd) != 0:
        raise ValueError("lbnd must be a scalar.")
    if jax.numpy.ndim(scl) != 0:
        raise ValueError("scl must be a scalar.")

    if m == 0:
        return c

    c = jax.numpy.moveaxis(c, axis, 0)
    k = jax.numpy.array(list(k) + [0] * (m - len(k)), ndmin=1)

    for i in range(m):
        n = len(c)
        c *= scl
        tmp = jax.numpy.empty((n + 1,) + c.shape[1:], dtype=c.dtype)
        tmp = tmp.at[0].set(c[0] * 0)
        tmp = tmp.at[1].set(c[0] / 2)
        j = jax.numpy.arange(1, n)
        tmp = tmp.at[j + 1].set((c[j].T / (2 * (j + 1))).T)
        tmp = tmp.at[0].add(k[i] - hermval(lbnd, tmp))
        c = tmp
    c = jax.numpy.moveaxis(c, 0, axis)
    return c


def hermval(x, c, tensor=True):
    c = as_series(c)
    x = jax.numpy.asarray(x)
    if tensor:
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
        c0 = c[-2] * jax.numpy.ones_like(x)
        c1 = c[-1] * jax.numpy.ones_like(x)

        def body(i, val):
            c0, c1, nd = val
            tmp = c0
            nd = nd - 1
            c0 = c[-i] - c1 * (2 * (nd - 1))
            c1 = tmp + c1 * x2
            return c0, c1, nd

        c0, c1, _ = jax.lax.fori_loop(3, len(c) + 1, body, (c0, c1, nd))

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
    if deg < 0:
        raise ValueError("deg must be non-negative")

    x = jax.numpy.array(x, ndmin=1)
    dims = (deg + 1,) + x.shape
    dtyp = jax.numpy.promote_types(x.dtype, jax.numpy.array(0.0).dtype)
    x = x.astype(dtyp)
    v = jax.numpy.empty(dims, dtype=dtyp)
    v = v.at[0].set(jax.numpy.ones_like(x))
    if deg > 0:
        x2 = x * 2
        v = v.at[1].set(x2)

        def body(i, v):
            return v.at[i].set(v[i - 1] * x2 - v[i - 2] * (2 * (i - 1)))

        v = jax.lax.fori_loop(2, deg + 1, body, v)

    return jax.numpy.moveaxis(v, 0, -1)


def hermvander2d(x, y, deg):
    return _vander_nd_flat((hermvander, hermvander), (x, y), deg)


def hermvander3d(x, y, z, deg):
    return _vander_nd_flat((hermvander, hermvander, hermvander), (x, y, z), deg)


def hermfit(x, y, deg, rcond=None, full=False, w=None):
    return _fit(hermvander, x, y, deg, rcond, full, w)


def hermcompanion(c):
    c = as_series(c)
    if len(c) < 2:
        raise ValueError("Series must have maximum degree of at least 1.")
    if len(c) == 2:
        return jax.numpy.array([[-0.5 * c[0] / c[1]]])

    n = len(c) - 1
    mat = jax.numpy.zeros((n, n), dtype=c.dtype)
    scl = jax.numpy.hstack(
        (1.0, 1.0 / jax.numpy.sqrt(2.0 * jax.numpy.arange(n - 1, 0, -1)))
    )
    scl = jax.numpy.cumprod(scl)[::-1]
    shp = mat.shape
    mat = mat.flatten()
    mat = mat.at[1 :: n + 1].set(jax.numpy.sqrt(0.5 * jax.numpy.arange(1, n)))
    mat = mat.at[n :: n + 1].set(jax.numpy.sqrt(0.5 * jax.numpy.arange(1, n)))
    mat = mat.reshape(shp)
    mat = mat.at[:, -1].add(-scl * c[:-1] / (2.0 * c[-1]))
    return mat


def hermroots(c):
    c = as_series(c)
    if len(c) <= 1:
        return jax.numpy.array([], dtype=c.dtype)
    if len(c) == 2:
        return jax.numpy.array([-0.5 * c[0] / c[1]])

    m = hermcompanion(c)[::-1, ::-1]
    r = jax.numpy.linalg.eigvals(m)
    r = jax.numpy.sort(r)
    return r


def _normed_hermite_n(x, n):
    def truefun():
        return jax.numpy.full(x.shape, 1 / jax.numpy.sqrt(jax.numpy.sqrt(jax.numpy.pi)))

    def falsefun():
        c0 = jax.numpy.zeros_like(x)
        c1 = jax.numpy.ones_like(x) / jax.numpy.sqrt(jax.numpy.sqrt(jax.numpy.pi))
        nd = jax.numpy.array(n).astype(float)

        def body(i, val):
            c0, c1, nd = val
            tmp = c0
            c0 = -c1 * jax.numpy.sqrt((nd - 1.0) / nd)
            c1 = tmp + c1 * x * jax.numpy.sqrt(2.0 / nd)
            nd = nd - 1.0
            return c0, c1, nd

        c0, c1, _ = jax.lax.fori_loop(0, n - 1, body, (c0, c1, nd))
        return c0 + c1 * x * jax.numpy.sqrt(2)

    return jax.lax.cond(n == 0, truefun, falsefun)


def hermgauss(deg):
    deg = int(deg)
    if deg <= 0:
        raise ValueError("deg must be a positive integer")

    c = jax.numpy.zeros(deg + 1).at[-1].set(1)
    m = hermcompanion(c)
    x = jax.numpy.linalg.eigvalsh(m)

    dy = _normed_hermite_n(x, deg)
    df = _normed_hermite_n(x, deg - 1) * jax.numpy.sqrt(2 * deg)
    x -= dy / df

    fm = _normed_hermite_n(x, deg - 1)
    fm /= jax.numpy.abs(fm).max()
    w = 1 / (fm * fm)

    w = (w + w[::-1]) / 2
    x = (x - x[::-1]) / 2

    w *= jax.numpy.sqrt(jax.numpy.pi) / w.sum()

    return x, w


def hermweight(x):
    w = jax.numpy.exp(-(x**2))
    return w


def poly2herme(pol):
    pol = as_series(pol)
    deg = len(pol) - 1

    res = jax.numpy.zeros_like(pol)

    def body(i, res):
        k = deg - i
        res = hermeadd(hermemulx(res, mode="same"), pol[k])
        return res

    res = jax.lax.fori_loop(0, deg + 1, body, res)
    return res


def herme2poly(c):
    from .polynomial import polyadd, polymulx, polysub

    c = as_series(c)
    n = len(c)
    if n == 1:
        return c
    if n == 2:
        return c
    else:
        c0 = jax.numpy.zeros_like(c).at[0].set(c[-2])
        c1 = jax.numpy.zeros_like(c).at[0].set(c[-1])

        def body(k, c0c1):
            i = n - 1 - k
            c0, c1 = c0c1
            tmp = c0
            c0 = polysub(c[i - 2], c1 * (i - 1))
            c1 = polyadd(tmp, polymulx(c1, "same"))
            return c0, c1

        c0, c1 = jax.lax.fori_loop(0, n - 2, body, (c0, c1))

        return polyadd(c0, polymulx(c1, "same"))


def hermeline(off, scl):
    return jax.numpy.array([off, scl])


def hermefromroots(roots):
    return _fromroots(hermeline, hermemul, roots)


def hermeadd(c1, c2):
    return _add(c1, c2)


def hermesub(c1, c2):
    return _sub(c1, c2)


def hermemulx(c, mode="full"):
    c = as_series(c)
    prd = jax.numpy.zeros(len(c) + 1, dtype=c.dtype)
    prd = prd.at[1].set(c[0])

    i = jax.numpy.arange(1, len(c))

    prd = prd.at[i + 1].set(c[i])
    prd = prd.at[i - 1].add(c[i] * i)

    if mode == "same":
        prd = prd[: len(c)]
    return prd


def hermemul(c1, c2, mode="full"):
    c1, c2 = as_series(c1, c2)
    lc1, lc2 = len(c1), len(c2)
    if lc1 > lc2:
        c = c2
        xs = c1
    else:
        c = c1
        xs = c2

    if len(c) == 1:
        c0 = hermeadd(jax.numpy.zeros(lc1 + lc2 - 1), c[0] * xs)
        c1 = jax.numpy.zeros(lc1 + lc2 - 1)
    elif len(c) == 2:
        c0 = hermeadd(jax.numpy.zeros(lc1 + lc2 - 1), c[0] * xs)
        c1 = hermeadd(jax.numpy.zeros(lc1 + lc2 - 1), c[1] * xs)
    else:
        nd = len(c)
        c0 = hermeadd(jax.numpy.zeros(lc1 + lc2 - 1), c[-2] * xs)
        c1 = hermeadd(jax.numpy.zeros(lc1 + lc2 - 1), c[-1] * xs)

        def body(i, val):
            c0, c1, nd = val
            tmp = c0
            nd = nd - 1
            c0 = hermesub(c[-i] * xs, c1 * (nd - 1))
            c1 = hermeadd(tmp, hermemulx(c1, "same"))
            return c0, c1, nd

        c0, c1, _ = jax.lax.fori_loop(3, len(c) + 1, body, (c0, c1, nd))

    ret = hermeadd(c0, hermemulx(c1, "same"))
    if mode == "same":
        ret = ret[: max(lc1, lc2)]
    return ret


def hermediv(c1, c2):
    return _div(hermemul, c1, c2)


def hermepow(c, pow, maxpower=16):
    return _pow(hermemul, c, pow, maxpower)


def hermeder(c, m=1, scl=1, axis=0):
    if m < 0:
        raise ValueError("The order of derivation must be non-negative")

    c = as_series(c)

    if m == 0:
        return c

    c = jax.numpy.moveaxis(c, axis, 0)
    n = len(c)
    if m >= n:
        c = jax.numpy.zeros_like(c[:1])
    else:
        for _ in range(m):
            n = n - 1
            c *= scl
            der = jax.numpy.empty((n,) + c.shape[1:], dtype=c.dtype)
            j = jax.numpy.arange(n, 0, -1)
            der = der.at[j - 1].set((j * c[j].T).T)
            c = der
    c = jax.numpy.moveaxis(c, 0, axis)
    return c


def hermeint(c, m=1, k=None, lbnd=0, scl=1, axis=0):
    if k is None:
        k = []
    c = as_series(c)
    lbnd, scl = map(jax.numpy.asarray, (lbnd, scl))

    if not jax.numpy.iterable(k):
        k = [k]
    if len(k) > m:
        raise ValueError("Too many integration constants")
    if jax.numpy.ndim(lbnd) != 0:
        raise ValueError("lbnd must be a scalar.")
    if jax.numpy.ndim(scl) != 0:
        raise ValueError("scl must be a scalar.")

    if m == 0:
        return c

    c = jax.numpy.moveaxis(c, axis, 0)
    k = jax.numpy.array(list(k) + [0] * (m - len(k)), ndmin=1)

    for i in range(m):
        n = len(c)
        c *= scl
        tmp = jax.numpy.empty((n + 1,) + c.shape[1:], dtype=c.dtype)
        tmp = tmp.at[0].set(c[0] * 0)
        tmp = tmp.at[1].set(c[0])
        j = jax.numpy.arange(1, n)
        tmp = tmp.at[j + 1].set((c[j].T / (j + 1)).T)
        tmp = tmp.at[0].add(k[i] - hermeval(lbnd, tmp))
        c = tmp
    c = jax.numpy.moveaxis(c, 0, axis)
    return c


def hermeval(x, c, tensor=True):
    c = as_series(c)
    x = jax.numpy.asarray(x)
    if tensor:
        c = c.reshape(c.shape + (1,) * x.ndim)

    if len(c) == 1:
        c0 = c[0]
        c1 = 0
    elif len(c) == 2:
        c0 = c[0]
        c1 = c[1]
    else:
        nd = len(c)
        c0 = c[-2] * jax.numpy.ones_like(x)
        c1 = c[-1] * jax.numpy.ones_like(x)

        def body(i, val):
            c0, c1, nd = val
            tmp = c0
            nd = nd - 1
            c0 = c[-i] - c1 * (nd - 1)
            c1 = tmp + c1 * x
            return c0, c1, nd

        c0, c1, _ = jax.lax.fori_loop(3, len(c) + 1, body, (c0, c1, nd))

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
    if deg < 0:
        raise ValueError("deg must be non-negative")

    x = jax.numpy.array(x, ndmin=1)
    dims = (deg + 1,) + x.shape
    dtyp = jax.numpy.promote_types(x.dtype, jax.numpy.array(0.0).dtype)
    x = x.astype(dtyp)
    v = jax.numpy.empty(dims, dtype=dtyp)
    v = v.at[0].set(jax.numpy.ones_like(x))
    if deg > 0:
        v = v.at[1].set(x)

        def body(i, v):
            return v.at[i].set(v[i - 1] * x - v[i - 2] * (i - 1))

        v = jax.lax.fori_loop(2, deg + 1, body, v)

    return jax.numpy.moveaxis(v, 0, -1)


def hermevander2d(x, y, deg):
    return _vander_nd_flat((hermevander, hermevander), (x, y), deg)


def hermevander3d(x, y, z, deg):
    return _vander_nd_flat((hermevander, hermevander, hermevander), (x, y, z), deg)


def hermefit(x, y, deg, rcond=None, full=False, w=None):
    return _fit(hermevander, x, y, deg, rcond, full, w)


def hermecompanion(c):
    c = as_series(c)
    if len(c) < 2:
        raise ValueError("Series must have maximum degree of at least 1.")
    if len(c) == 2:
        return jax.numpy.array([[-c[0] / c[1]]])

    n = len(c) - 1
    mat = jax.numpy.zeros((n, n), dtype=c.dtype)
    scl = jax.numpy.hstack((1.0, 1.0 / jax.numpy.sqrt(jax.numpy.arange(n - 1, 0, -1))))
    scl = jax.numpy.cumprod(scl)[::-1]
    shp = mat.shape
    mat = mat.flatten()
    mat = mat.at[1 :: n + 1].set(jax.numpy.sqrt(jax.numpy.arange(1, n)))
    mat = mat.at[n :: n + 1].set(jax.numpy.sqrt(jax.numpy.arange(1, n)))
    mat = mat.reshape(shp)
    mat = mat.at[:, -1].add(-scl * c[:-1] / c[-1])
    return mat


def hermeroots(c):
    c = as_series(c)
    if len(c) <= 1:
        return jax.numpy.array([], dtype=c.dtype)
    if len(c) == 2:
        return jax.numpy.array([-c[0] / c[1]])

    m = hermecompanion(c)[::-1, ::-1]
    r = jax.numpy.linalg.eigvals(m)
    r = jax.numpy.sort(r)
    return r


def _normed_hermite_e_n(x, n):
    def truefun():
        return jax.numpy.full(
            x.shape, 1 / jax.numpy.sqrt(jax.numpy.sqrt(2 * jax.numpy.pi))
        )

    def falsefun():
        c0 = jax.numpy.zeros_like(x)
        c1 = jax.numpy.ones_like(x) / jax.numpy.sqrt(jax.numpy.sqrt(2 * jax.numpy.pi))
        nd = jax.numpy.array(n).astype(float)

        def body(i, val):
            c0, c1, nd = val
            tmp = c0
            c0 = -c1 * jax.numpy.sqrt((nd - 1.0) / nd)
            c1 = tmp + c1 * x * jax.numpy.sqrt(1.0 / nd)
            nd = nd - 1.0
            return c0, c1, nd

        c0, c1, _ = jax.lax.fori_loop(0, n - 1, body, (c0, c1, nd))
        return c0 + c1 * x

    return jax.lax.cond(n == 0, truefun, falsefun)


def hermegauss(deg):
    deg = int(deg)
    if deg <= 0:
        raise ValueError("deg must be a positive integer")

    c = jax.numpy.zeros(deg + 1).at[-1].set(1)
    m = hermecompanion(c)
    x = jax.numpy.linalg.eigvalsh(m)

    dy = _normed_hermite_e_n(x, deg)
    df = _normed_hermite_e_n(x, deg - 1) * jax.numpy.sqrt(deg)
    x -= dy / df

    fm = _normed_hermite_e_n(x, deg - 1)
    fm /= jax.numpy.abs(fm).max()
    w = 1 / (fm * fm)

    w = (w + w[::-1]) / 2
    x = (x - x[::-1]) / 2

    w *= jax.numpy.sqrt(2 * jax.numpy.pi) / w.sum()

    return x, w


def hermeweight(x):
    w = jax.numpy.exp(-0.5 * x**2)
    return w


def poly2lag(pol):
    pol = as_series(pol)
    res = jax.numpy.zeros_like(pol)

    def body(i, res):
        res = lagadd(lagmulx(res, mode="same"), pol[::-1][i])
        return res

    res = jax.lax.fori_loop(0, len(pol), body, res)
    return res


def lag2poly(c):
    from .polynomial import polyadd, polymulx, polysub

    c = as_series(c)
    n = len(c)
    if n == 1:
        return c
    else:
        c0 = jax.numpy.zeros_like(c).at[0].set(c[-2])
        c1 = jax.numpy.zeros_like(c).at[0].set(c[-1])

        def body(k, c0c1):
            i = n - 1 - k
            c0, c1 = c0c1
            tmp = c0
            c0 = polysub(c[i - 2], (c1 * (i - 1)) / i)
            c1 = polyadd(tmp, polysub((2 * i - 1) * c1, polymulx(c1, "same")) / i)
            return c0, c1

        c0, c1 = jax.lax.fori_loop(0, n - 2, body, (c0, c1))

        return polyadd(c0, polysub(c1, polymulx(c1, "same")))


def lagline(off, scl):
    return jax.numpy.array([off + scl, -scl])


def lagfromroots(roots):
    return _fromroots(lagline, lagmul, roots)


def lagadd(c1, c2):
    return _add(c1, c2)


def lagsub(c1, c2):
    return _sub(c1, c2)


def lagmulx(c, mode="full"):
    c = as_series(c)

    prd = jax.numpy.zeros(len(c) + 1, dtype=c.dtype)
    prd = prd.at[0].set(c[0])
    prd = prd.at[1].set(-c[0])

    i = jax.numpy.arange(1, len(c))

    prd = prd.at[i + 1].set(-c[i] * (i + 1))
    prd = prd.at[i].add(c[i] * (2 * i + 1))
    prd = prd.at[i - 1].add(-c[i] * i)

    if mode == "same":
        prd = prd[: len(c)]
    return prd


def lagmul(c1, c2, mode="full"):
    c1, c2 = as_series(c1, c2)
    lc1, lc2 = len(c1), len(c2)

    if lc1 > lc2:
        c = c2
        xs = c1
    else:
        c = c1
        xs = c2

    if len(c) == 1:
        c0 = lagadd(jax.numpy.zeros(lc1 + lc2 - 1), c[0] * xs)
        c1 = jax.numpy.zeros(lc1 + lc2 - 1)
    elif len(c) == 2:
        c0 = lagadd(jax.numpy.zeros(lc1 + lc2 - 1), c[0] * xs)
        c1 = lagadd(jax.numpy.zeros(lc1 + lc2 - 1), c[1] * xs)
    else:
        nd = len(c)
        c0 = lagadd(jax.numpy.zeros(lc1 + lc2 - 1), c[-2] * xs)
        c1 = lagadd(jax.numpy.zeros(lc1 + lc2 - 1), c[-1] * xs)

        def body(i, val):
            c0, c1, nd = val
            tmp = c0
            nd = nd - 1
            c0 = lagsub(c[-i] * xs, (c1 * (nd - 1)) / nd)
            c1 = lagadd(tmp, lagsub((2 * nd - 1) * c1, lagmulx(c1, "same")) / nd)
            return c0, c1, nd

        c0, c1, _ = jax.lax.fori_loop(3, len(c) + 1, body, (c0, c1, nd))

    ret = lagadd(c0, lagsub(c1, lagmulx(c1, "same")))
    if mode == "same":
        ret = ret[: max(lc1, lc2)]
    return ret


def lagdiv(c1, c2):
    return _div(lagmul, c1, c2)


def lagpow(c, pow, maxpower=16):
    return _pow(lagmul, c, pow, maxpower)


def lagder(c, m=1, scl=1, axis=0):
    if m < 0:
        raise ValueError("The order of derivation must be non-negative")

    c = as_series(c)

    if m == 0:
        return c

    c = jax.numpy.moveaxis(c, axis, 0)
    n = len(c)
    if m >= n:
        c = jax.numpy.zeros_like(c[:1])
    else:
        for _ in range(m):
            n = n - 1
            c *= scl
            der = jax.numpy.empty((n,) + c.shape[1:], dtype=c.dtype)

            def body(k, der_c, n=n):
                j = n - k
                der, c = der_c
                der = der.at[j - 1].set(-c[j])
                c = c.at[j - 1].add(c[j])
                return der, c

            der, c = jax.lax.fori_loop(0, n - 1, body, (der, c))
            der = der.at[0].set(-c[1])
            c = der

    c = jax.numpy.moveaxis(c, 0, axis)
    return c


def lagint(c, m=1, k=None, lbnd=0, scl=1, axis=0):
    if k is None:
        k = []
    c = as_series(c)
    lbnd, scl = map(jax.numpy.asarray, (lbnd, scl))

    if not jax.numpy.iterable(k):
        k = [k]
    if len(k) > m:
        raise ValueError("Too many integration constants")
    if jax.numpy.ndim(lbnd) != 0:
        raise ValueError("lbnd must be a scalar.")
    if jax.numpy.ndim(scl) != 0:
        raise ValueError("scl must be a scalar.")

    if m == 0:
        return c

    c = jax.numpy.moveaxis(c, axis, 0)
    k = jax.numpy.array(list(k) + [0] * (m - len(k)), ndmin=1)

    for i in range(m):
        n = len(c)
        c *= scl
        tmp = jax.numpy.empty((n + 1,) + c.shape[1:], dtype=c.dtype)
        tmp = tmp.at[0].set(c[0])
        tmp = tmp.at[1].set(-c[0])
        j = jax.numpy.arange(1, n)
        tmp = tmp.at[j].add(c[j])
        tmp = tmp.at[j + 1].add(-c[j])
        tmp = tmp.at[0].add(k[i] - lagval(lbnd, tmp))
        c = tmp

    c = jax.numpy.moveaxis(c, 0, axis)
    return c


def lagval(x, c, tensor=True):
    c = as_series(c)
    x = jax.numpy.asarray(x)
    if tensor:
        c = c.reshape(c.shape + (1,) * x.ndim)

    if len(c) == 1:
        c0 = c[0]
        c1 = 0
    elif len(c) == 2:
        c0 = c[0]
        c1 = c[1]
    else:
        nd = len(c)
        c0 = c[-2] * jax.numpy.ones_like(x)
        c1 = c[-1] * jax.numpy.ones_like(x)

        def body(i, val):
            c0, c1, nd = val
            tmp = c0
            nd = nd - 1
            c0 = c[-i] - (c1 * (nd - 1)) / nd
            c1 = tmp + (c1 * ((2 * nd - 1) - x)) / nd
            return c0, c1, nd

        c0, c1, _ = jax.lax.fori_loop(3, len(c) + 1, body, (c0, c1, nd))

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
    if deg < 0:
        raise ValueError("deg must be non-negative")

    x = jax.numpy.array(x, ndmin=1)
    dims = (deg + 1,) + x.shape
    dtyp = jax.numpy.promote_types(x.dtype, jax.numpy.array(0.0).dtype)
    x = x.astype(dtyp)
    v = jax.numpy.empty(dims, dtype=dtyp)
    v = v.at[0].set(jax.numpy.ones_like(x))
    if deg > 0:
        v = v.at[1].set(1 - x)

        def body(i, v):
            return v.at[i].set((v[i - 1] * (2 * i - 1 - x) - v[i - 2] * (i - 1)) / i)

        v = jax.lax.fori_loop(2, deg + 1, body, v)

    return jax.numpy.moveaxis(v, 0, -1)


def lagvander2d(x, y, deg):
    return _vander_nd_flat((lagvander, lagvander), (x, y), deg)


def lagvander3d(x, y, z, deg):
    return _vander_nd_flat((lagvander, lagvander, lagvander), (x, y, z), deg)


def lagfit(x, y, deg, rcond=None, full=False, w=None):
    return _fit(lagvander, x, y, deg, rcond, full, w)


def lagcompanion(c):
    c = as_series(c)
    if len(c) < 2:
        raise ValueError("Series must have maximum degree of at least 1.")
    if len(c) == 2:
        return jax.numpy.array([[1 + c[0] / c[1]]])

    n = len(c) - 1
    mat = jax.numpy.zeros((n, n), dtype=c.dtype).flatten()
    mat = mat.at[1 :: n + 1].set(-jax.numpy.arange(1, n))
    mat = mat.at[0 :: n + 1].set(2.0 * jax.numpy.arange(n) + 1.0)
    mat = mat.at[n :: n + 1].set(-jax.numpy.arange(1, n))
    mat = mat.reshape((n, n))
    mat = mat.at[:, -1].add((c[:-1] / c[-1]) * n)
    return mat


def lagroots(c):
    c = as_series(c)
    if len(c) <= 1:
        return jax.numpy.array([], dtype=c.dtype)
    if len(c) == 2:
        return jax.numpy.array([1 + c[0] / c[1]])

    m = lagcompanion(c)[::-1, ::-1]
    r = jax.numpy.linalg.eigvals(m)
    r = jax.numpy.sort(r)
    return r


def laggauss(deg):
    deg = int(deg)
    if deg <= 0:
        raise ValueError("deg must be a positive integer")

    c = jax.numpy.zeros(deg + 1).at[-1].set(1)
    m = lagcompanion(c)
    x = jax.numpy.linalg.eigvalsh(m)

    dy = lagval(x, c)
    df = lagval(x, lagder(c))
    x -= dy / df

    fm = lagval(x, c[1:])
    fm /= jax.numpy.abs(fm).max()
    df /= jax.numpy.abs(df).max()
    w = 1 / (fm * df)

    w /= w.sum()

    return x, w


def lagweight(x):
    w = jax.numpy.exp(-x)
    return w


def poly2leg(pol):
    pol = as_series(pol)
    deg = len(pol) - 1

    res = jax.numpy.zeros_like(pol)

    def body(i, res):
        k = deg - i
        res = legadd(legmulx(res, mode="same"), pol[k])
        return res

    res = jax.lax.fori_loop(0, deg + 1, body, res)
    return res


def leg2poly(c):
    from .polynomial import polyadd, polymulx, polysub

    c = as_series(c)
    n = len(c)
    if n < 3:
        return c
    else:
        c0 = jax.numpy.zeros_like(c).at[0].set(c[-2])
        c1 = jax.numpy.zeros_like(c).at[0].set(c[-1])

        def body(k, c0c1):
            i = n - 1 - k
            c0, c1 = c0c1
            tmp = c0
            c0 = polysub(c[i - 2], c1 * (i - 1) / i)
            c1 = polyadd(tmp, polymulx(c1, "same") * (2 * i - 1) / i)
            return c0, c1

        c0, c1 = jax.lax.fori_loop(0, n - 2, body, (c0, c1))

        return polyadd(c0, polymulx(c1, "same"))


def legline(off, scl):
    return jax.numpy.array([off, scl])


def legfromroots(roots):
    return _fromroots(legline, legmul, roots)


def legadd(c1, c2):
    return _add(c1, c2)


def legsub(c1, c2):
    return _sub(c1, c2)


def legmulx(c, mode="full"):
    c = as_series(c)
    prd = jax.numpy.zeros(len(c) + 1, dtype=c.dtype)
    prd = prd.at[1].set(c[0])

    def body(i, prd):
        j = i + 1
        k = i - 1
        s = i + j
        prd = prd.at[j].set((c[i] * j) / s)
        prd = prd.at[k].add((c[i] * i) / s)
        return prd

    prd = jax.lax.fori_loop(1, len(c), body, prd)

    if mode == "same":
        prd = prd[: len(c)]
    return prd


def legmul(c1, c2, mode="full"):
    c1, c2 = as_series(c1, c2)
    lc1, lc2 = len(c1), len(c2)
    if lc1 > lc2:
        c = c2
        xs = c1
    else:
        c = c1
        xs = c2

    if len(c) == 1:
        c0 = legadd(jax.numpy.zeros(lc1 + lc2 - 1), c[0] * xs)
        c1 = jax.numpy.zeros(lc1 + lc2 - 1)
    elif len(c) == 2:
        c0 = legadd(jax.numpy.zeros(lc1 + lc2 - 1), c[0] * xs)
        c1 = legadd(jax.numpy.zeros(lc1 + lc2 - 1), c[1] * xs)
    else:
        nd = len(c)
        c0 = legadd(jax.numpy.zeros(lc1 + lc2 - 1), c[-2] * xs)
        c1 = legadd(jax.numpy.zeros(lc1 + lc2 - 1), c[-1] * xs)

        def body(i, val):
            c0, c1, nd = val
            tmp = c0
            nd = nd - 1
            c0 = legsub(c[-i] * xs, (c1 * (nd - 1)) / nd)
            c1 = legadd(tmp, (legmulx(c1, "same") * (2 * nd - 1)) / nd)
            return c0, c1, nd

        c0, c1, _ = jax.lax.fori_loop(3, len(c) + 1, body, (c0, c1, nd))

    ret = legadd(c0, legmulx(c1, "same"))
    if mode == "same":
        ret = ret[: max(lc1, lc2)]
    return ret


def legdiv(c1, c2):
    return _div(legmul, c1, c2)


def legpow(c, pow, maxpower=16):
    return _pow(legmul, c, pow, maxpower)


def legder(c, m=1, scl=1, axis=0):
    if m < 0:
        raise ValueError("The order of derivation must be non-negative")

    c = as_series(c)

    if m == 0:
        return c

    c = jax.numpy.moveaxis(c, axis, 0)
    n = len(c)
    if m >= n:
        c = jax.numpy.zeros_like(c[:1])
    else:
        for _ in range(m):
            n = n - 1
            c *= scl
            der = jax.numpy.empty((n,) + c.shape[1:], dtype=c.dtype)

            def body(k, der_c, n=n):
                j = n - k
                der, c = der_c
                der = der.at[j - 1].set((2 * j - 1) * c[j])
                c = c.at[j - 2].add(c[j])
                return der, c

            der, c = jax.lax.fori_loop(0, n - 2, body, (der, c))
            if n > 1:
                der = der.at[1].set(3 * c[2])
            der = der.at[0].set(c[1])
            c = der

    c = jax.numpy.moveaxis(c, 0, axis)
    return c


def legint(c, m=1, k=None, lbnd=0, scl=1, axis=0):
    if k is None:
        k = []
    c = as_series(c)
    lbnd, scl = map(jax.numpy.asarray, (lbnd, scl))

    if not jax.numpy.iterable(k):
        k = [k]
    if len(k) > m:
        raise ValueError("Too many integration constants")
    if jax.numpy.ndim(lbnd) != 0:
        raise ValueError("lbnd must be a scalar.")
    if jax.numpy.ndim(scl) != 0:
        raise ValueError("scl must be a scalar.")

    if m == 0:
        return c

    c = jax.numpy.moveaxis(c, axis, 0)
    k = jax.numpy.array(list(k) + [0] * (m - len(k)), ndmin=1)

    for i in range(m):
        n = len(c)
        c *= scl
        tmp = jax.numpy.empty((n + 1,) + c.shape[1:], dtype=c.dtype)
        tmp = tmp.at[0].set(c[0] * 0)
        tmp = tmp.at[1].set(c[0])
        if n > 1:
            tmp = tmp.at[2].set(c[1] / 3)
        j = jax.numpy.arange(2, n)
        t = (c[j].T / (2 * j + 1)).T
        tmp = tmp.at[j + 1].set(t)
        tmp = tmp.at[j - 1].add(-t)
        tmp = tmp.at[0].add(k[i] - legval(lbnd, tmp))
        c = tmp
    c = jax.numpy.moveaxis(c, 0, axis)
    return c


def legval(x, c, tensor=True):
    c = as_series(c)
    x = jax.numpy.asarray(x)
    if tensor:
        c = c.reshape(c.shape + (1,) * x.ndim)

    if len(c) == 1:
        c0 = c[0]
        c1 = 0
    elif len(c) == 2:
        c0 = c[0]
        c1 = c[1]
    else:
        nd = len(c)
        c0 = c[-2] * jax.numpy.ones_like(x)
        c1 = c[-1] * jax.numpy.ones_like(x)

        def body(i, val):
            c0, c1, nd = val
            tmp = c0
            nd = nd - 1
            c0 = c[-i] - (c1 * (nd - 1)) / nd
            c1 = tmp + (c1 * x * (2 * nd - 1)) / nd
            return c0, c1, nd

        c0, c1, _ = jax.lax.fori_loop(3, len(c) + 1, body, (c0, c1, nd))

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
    if deg < 0:
        raise ValueError("deg must be non-negative")

    x = jax.numpy.array(x, ndmin=1)
    dims = (deg + 1,) + x.shape
    dtyp = jax.numpy.promote_types(x.dtype, jax.numpy.array(0.0).dtype)
    x = x.astype(dtyp)
    v = jax.numpy.empty(dims, dtype=dtyp)
    v = v.at[0].set(jax.numpy.ones_like(x))
    if deg > 0:
        v = v.at[1].set(x)

        def body(i, v):
            return v.at[i].set((v[i - 1] * x * (2 * i - 1) - v[i - 2] * (i - 1)) / i)

        v = jax.lax.fori_loop(2, deg + 1, body, v)

    return jax.numpy.moveaxis(v, 0, -1)


def legvander2d(x, y, deg):
    return _vander_nd_flat((legvander, legvander), (x, y), deg)


def legvander3d(x, y, z, deg):
    return _vander_nd_flat((legvander, legvander, legvander), (x, y, z), deg)


def legfit(x, y, deg, rcond=None, full=False, w=None):
    return _fit(legvander, x, y, deg, rcond, full, w)


def legcompanion(c):
    c = as_series(c)
    if len(c) < 2:
        raise ValueError("Series must have maximum degree of at least 1.")
    if len(c) == 2:
        return jax.numpy.array([[-c[0] / c[1]]])

    n = len(c) - 1
    mat = jax.numpy.zeros((n, n), dtype=c.dtype)
    scl = 1.0 / jax.numpy.sqrt(2 * jax.numpy.arange(n) + 1)
    shp = mat.shape
    mat = mat.flatten()
    mat = mat.at[1 :: n + 1].set(jax.numpy.arange(1, n) * scl[: n - 1] * scl[1:n])
    mat = mat.at[n :: n + 1].set(jax.numpy.arange(1, n) * scl[: n - 1] * scl[1:n])
    mat = mat.reshape(shp)
    mat = mat.at[:, -1].add(-(c[:-1] / c[-1]) * (scl / scl[-1]) * (n / (2 * n - 1)))
    return mat


def legroots(c):
    c = as_series(c)
    if len(c) <= 1:
        return jax.numpy.array([], dtype=c.dtype)
    if len(c) == 2:
        return jax.numpy.array([-c[0] / c[1]])

    m = legcompanion(c)[::-1, ::-1]
    r = jax.numpy.linalg.eigvals(m)
    r = jax.numpy.sort(r)
    return r


def leggauss(deg):
    deg = int(deg)
    if deg <= 0:
        raise ValueError("deg must be a positive integer")

    c = jax.numpy.zeros(deg + 1).at[-1].set(1)
    m = legcompanion(c)
    x = jax.numpy.linalg.eigvalsh(m)

    dy = legval(x, c)
    df = legval(x, legder(c))
    x -= dy / df

    fm = legval(x, c[1:])
    fm /= jax.numpy.abs(fm).max()
    df /= jax.numpy.abs(df).max()
    w = 1 / (fm * df)

    w = (w + w[::-1]) / 2
    x = (x - x[::-1]) / 2

    w *= 2.0 / w.sum()

    return x, w


def legweight(x):
    w = jax.numpy.ones_like(x)
    return w


def polyline(off, scl):
    return jax.numpy.array([off, scl])


def polyfromroots(roots):
    return _fromroots(polyline, polymul, roots)


def polyadd(c1, c2):
    return _add(c1, c2)


def polysub(c1, c2):
    return _sub(c1, c2)


def polymulx(c, mode="full"):
    c = as_series(c)
    prd = jax.numpy.zeros(len(c) + 1, dtype=c.dtype)
    prd = prd.at[1:].set(c)
    if mode == "same":
        prd = prd[: len(c)]
    return prd


def polymul(c1, c2, mode="full"):
    c1, c2 = as_series(c1, c2)
    ret = jax.numpy.convolve(c1, c2)
    if mode == "same":
        ret = ret[: max(len(c1), len(c2))]
    return ret


def polydiv(c1, c2):
    c1, c2 = as_series(c1, c2)
    return _div(polymul, c1, c2)


def polypow(c, pow, maxpower=16):
    return _pow(polymul, c, pow, maxpower)


def polyder(c, m=1, scl=1, axis=0):
    c = as_series(c)

    if m == 0:
        return c

    c = jax.numpy.moveaxis(c, axis, 0)
    n = c.shape[0]
    if m >= n:
        c = jax.numpy.zeros_like(c[:1])
    else:
        D = jax.numpy.arange(n)

        def body(i, c):
            c = (D * c.T).T
            c = jax.numpy.roll(c, -1, axis=0) * scl
            c = c.at[-1].set(0)
            return c

        c = jax.lax.fori_loop(0, m, body, c)
        c = c[:-m]

    c = jax.numpy.moveaxis(c, 0, axis)
    return c


def polyint(c, m=1, k=None, lbnd=0, scl=1, axis=0):
    if k is None:
        k = []
    c = as_series(c)
    lbnd, scl = map(jax.numpy.asarray, (lbnd, scl))
    if not jax.numpy.iterable(k):
        k = [k]
    if m < 0:
        raise ValueError("The order of integration must be non-negative")
    if len(k) > m:
        raise ValueError("Too many integration constants")
    if jax.numpy.ndim(lbnd) != 0:
        raise ValueError("lbnd must be a scalar.")
    if jax.numpy.ndim(scl) != 0:
        raise ValueError("scl must be a scalar.")

    if m == 0:
        return c

    k = jax.numpy.array(list(k) + [0] * (m - len(k)), ndmin=1)
    n = c.shape[axis]
    c = _pad_along_axis(c, (0, m), axis)
    c = jax.numpy.moveaxis(c, axis, 0)
    D = jax.numpy.arange(n + m) + 1

    def body(i, c):
        c *= scl
        c = (c.T / D).T  # broadcasting correctly
        c = jax.numpy.roll(c, 1, axis=0)
        c = c.at[0].set(0)
        offset = k[i] - polyval(lbnd, c)
        c = c.at[0].add(offset)
        return c

    c = jax.lax.fori_loop(0, m, body, c)

    c = jax.numpy.moveaxis(c, 0, axis)
    return c


def polyval(x, c, tensor=True):
    c = as_series(c)
    x = jax.numpy.asarray(x)
    if tensor:
        c = c.reshape(c.shape + (1,) * x.ndim)

    c0 = c[-1] + jax.numpy.zeros_like(x)

    def body(i, c0):
        c0 = c[-i] + c0 * x
        return c0

    c0 = jax.lax.fori_loop(2, len(c) + 1, body, c0)
    return c0


def polyvalfromroots(x, r, tensor=True):
    r = jax.numpy.array(r, ndmin=1)
    x = jax.numpy.asarray(x)
    if tensor:
        r = r.reshape(r.shape + (1,) * x.ndim)
    elif x.ndim >= r.ndim:
        raise ValueError("x.ndim must be < r.ndim when tensor == False")
    return jax.numpy.prod(x - r, axis=0)


def polyval2d(x, y, c):
    return _valnd(polyval, c, x, y)


def polygrid2d(x, y, c):
    return _gridnd(polyval, c, x, y)


def polyval3d(x, y, z, c):
    return _valnd(polyval, c, x, y, z)


def polygrid3d(x, y, z, c):
    return _gridnd(polyval, c, x, y, z)


def polyvander(x, deg):
    if deg < 0:
        raise ValueError("deg must be non-negative")

    x = jax.numpy.array(x, ndmin=1)
    dims = (deg + 1,) + x.shape
    dtyp = x.dtype
    v = jax.numpy.empty(dims, dtype=dtyp)
    v = v.at[0].set(jax.numpy.ones_like(x))

    def body(i, v):
        v = v.at[i].set(v[i - 1] * x)
        return v

    v = jax.lax.fori_loop(1, deg + 1, body, v)

    return jax.numpy.moveaxis(v, 0, -1)


def polyvander2d(x, y, deg):
    return _vander_nd_flat((polyvander, polyvander), (x, y), deg)


def polyvander3d(x, y, z, deg):
    return _vander_nd_flat((polyvander, polyvander, polyvander), (x, y, z), deg)


def polyfit(x, y, deg, rcond=None, full=False, w=None):
    return _fit(polyvander, x, y, deg, rcond, full, w)


def polycompanion(c):
    c = as_series(c)
    if len(c) < 2:
        raise ValueError("Series must have maximum degree of at least 1.")
    if len(c) == 2:
        return jax.numpy.array([[-c[0] / c[1]]])

    n = len(c) - 1
    mat = jax.numpy.zeros((n, n), dtype=c.dtype).flatten()
    mat = mat.at[n :: n + 1].set(1)
    mat = mat.reshape((n, n))
    mat = mat.at[:, -1].add(-c[:-1] / c[-1])
    return mat


def polyroots(c):
    c = as_series(c)
    if len(c) < 2:
        return jax.numpy.array([], dtype=c.dtype)
    if len(c) == 2:
        return jax.numpy.array([-c[0] / c[1]])

    m = polycompanion(c)[::-1, ::-1]
    r = jax.numpy.linalg.eigvals(m)
    r = jax.numpy.sort(r)
    return r


def trimseq(seq):
    if len(seq) == 0:
        return seq
    else:
        for i in range(len(seq) - 1, -1, -1):
            if seq[i] != 0:
                break
        return seq[: i + 1]


def as_series(*arrs, trim=False):
    arrays = tuple(jax.numpy.array(a, ndmin=1) for a in arrs)
    if trim:
        arrays = tuple(trimseq(a) for a in arrays)
    arrays = jax._src.numpy.util.promote_dtypes_inexact(*arrays)
    if len(arrays) == 1:
        return arrays[0]
    return tuple(arrays)


def trimcoef(c, tol=0):
    if tol < 0:
        raise ValueError("tol must be non-negative")

    c = as_series(c)
    [ind] = jax.numpy.nonzero(jax.numpy.abs(c) > tol)
    if len(ind) == 0:
        return c[:1] * 0
    else:
        return c[: ind[-1] + 1].copy()


def getdomain(x):
    x = jax.numpy.asarray(x)
    if jax.numpy.iscomplexobj(x):
        rmin, rmax = x.real.min(), x.real.max()
        imin, imax = x.imag.min(), x.imag.max()
        return jax.numpy.array(((rmin + 1j * imin), (rmax + 1j * imax)))
    else:
        return jax.numpy.array((x.min(), x.max()))


def mapparms(old, new):
    oldlen = old[1] - old[0]
    newlen = new[1] - new[0]
    off = (old[1] * new[0] - old[0] * new[1]) / oldlen
    scl = newlen / oldlen
    return off, scl


def mapdomain(x, old, new):
    x = jax.numpy.asarray(x)
    off, scl = mapparms(old, new)
    return off + scl * x


chebtrim = trimcoef
hermetrim = trimcoef
hermtrim = trimcoef
lagtrim = trimcoef
legtrim = trimcoef
polytrim = trimcoef

__all__ = [
    "as_series",
    "cheb2poly",
    "chebadd",
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
    "chebsub",
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
    "getdomain",
    "herm2poly",
    "hermadd",
    "hermcompanion",
    "hermder",
    "hermdiv",
    "hermdomain",
    "herme2poly",
    "hermeadd",
    "hermecompanion",
    "hermeder",
    "hermediv",
    "hermedomain",
    "hermefit",
    "hermefromroots",
    "hermegauss",
    "hermegrid2d",
    "hermegrid3d",
    "hermeint",
    "hermeline",
    "hermemul",
    "hermemulx",
    "hermeone",
    "hermepow",
    "hermeroots",
    "hermesub",
    "hermetrim",
    "hermeval",
    "hermeval2d",
    "hermeval3d",
    "hermevander",
    "hermevander2d",
    "hermevander3d",
    "hermeweight",
    "hermex",
    "hermezero",
    "hermfit",
    "hermfromroots",
    "hermgauss",
    "hermgrid2d",
    "hermgrid3d",
    "hermint",
    "hermline",
    "hermmul",
    "hermmulx",
    "hermone",
    "hermpow",
    "hermroots",
    "hermsub",
    "hermtrim",
    "hermval",
    "hermval2d",
    "hermval3d",
    "hermvander",
    "hermvander2d",
    "hermvander3d",
    "hermweight",
    "hermx",
    "hermzero",
    "lag2poly",
    "lagadd",
    "lagcompanion",
    "lagder",
    "lagdiv",
    "lagdomain",
    "lagfit",
    "lagfromroots",
    "laggauss",
    "laggrid2d",
    "laggrid3d",
    "lagint",
    "lagline",
    "lagmul",
    "lagmulx",
    "lagone",
    "lagpow",
    "lagroots",
    "lagsub",
    "lagtrim",
    "lagval",
    "lagval2d",
    "lagval3d",
    "lagvander",
    "lagvander2d",
    "lagvander3d",
    "lagweight",
    "lagx",
    "lagzero",
    "leg2poly",
    "legadd",
    "legcompanion",
    "legder",
    "legdiv",
    "legdomain",
    "legfit",
    "legfromroots",
    "leggauss",
    "leggrid2d",
    "leggrid3d",
    "legint",
    "legline",
    "legmul",
    "legmulx",
    "legone",
    "legpow",
    "legroots",
    "legsub",
    "legtrim",
    "legval",
    "legval2d",
    "legval3d",
    "legvander",
    "legvander2d",
    "legvander3d",
    "legweight",
    "legx",
    "legzero",
    "mapdomain",
    "mapparms",
    "poly2cheb",
    "poly2herm",
    "poly2herme",
    "poly2lag",
    "poly2leg",
    "polyadd",
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
    "trimcoef",
    "trimseq",
]
