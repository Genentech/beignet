import jax
import jax.numpy

from . import polyutils

__all__ = [
    "chebzero",
    "chebone",
    "chebx",
    "chebdomain",
    "chebline",
    "chebadd",
    "chebsub",
    "chebmulx",
    "chebmul",
    "chebdiv",
    "chebpow",
    "chebval",
    "chebder",
    "chebint",
    "cheb2poly",
    "poly2cheb",
    "chebfromroots",
    "chebvander",
    "chebfit",
    "chebtrim",
    "chebroots",
    "chebpts1",
    "chebpts2",
    "chebval2d",
    "chebval3d",
    "chebgrid2d",
    "chebgrid3d",
    "chebvander2d",
    "chebvander3d",
    "chebcompanion",
    "chebgauss",
    "chebweight",
    "chebinterpolate",
]

chebtrim = polyutils.trimcoef


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
    pol = polyutils.as_series(pol)
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

    c = polyutils.as_series(c)
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


chebdomain = jax.numpy.array([-1, 1])


chebzero = jax.numpy.array([0])


chebone = jax.numpy.array([1])


chebx = jax.numpy.array([0, 1])


def chebline(off, scl):
    return jax.numpy.array([off, scl])


def chebfromroots(roots):
    return polyutils._fromroots(chebline, chebmul, roots)


def chebadd(c1, c2):
    return polyutils._add(c1, c2)


def chebsub(c1, c2):
    return polyutils._sub(c1, c2)


def chebmulx(c, mode="full"):
    c = polyutils.as_series(c)
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
    c1, c2 = polyutils.as_series(c1, c2)
    z1 = _cseries_to_zseries(c1)
    z2 = _cseries_to_zseries(c2)
    prd = _zseries_mul(z1, z2, mode=mode)
    ret = _zseries_to_cseries(prd)
    if mode == "same":
        ret = ret[: max(len(c1), len(c2))]

    return ret


def chebdiv(c1, c2):
    return polyutils._div(chebmul, c1, c2)


def chebpow(c, pow, maxpower=16):
    c = polyutils.as_series(c)
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

    c = polyutils.as_series(c)

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
    c = polyutils.as_series(c)
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
    c = polyutils.as_series(c)
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
    return polyutils._valnd(chebval, c, x, y)


def chebgrid2d(x, y, c):
    return polyutils._gridnd(chebval, c, x, y)


def chebval3d(x, y, z, c):
    return polyutils._valnd(chebval, c, x, y, z)


def chebgrid3d(x, y, z, c):
    return polyutils._gridnd(chebval, c, x, y, z)


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
    return polyutils._vander_nd_flat((chebvander, chebvander), (x, y), deg)


def chebvander3d(x, y, z, deg):
    return polyutils._vander_nd_flat(
        (chebvander, chebvander, chebvander), (x, y, z), deg
    )


def chebfit(x, y, deg, rcond=None, full=False, w=None):
    return polyutils._fit(chebvander, x, y, deg, rcond, full, w)


def chebcompanion(c):
    c = polyutils.as_series(c)
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
    c = polyutils.as_series(c)
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
