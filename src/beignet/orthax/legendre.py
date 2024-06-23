import jax
import jax.numpy

from . import polyutils

__all__ = [
    "legzero",
    "legone",
    "legx",
    "legdomain",
    "legline",
    "legadd",
    "legsub",
    "legmulx",
    "legmul",
    "legdiv",
    "legpow",
    "legval",
    "legder",
    "legint",
    "leg2poly",
    "poly2leg",
    "legfromroots",
    "legvander",
    "legfit",
    "legtrim",
    "legroots",
    "legval2d",
    "legval3d",
    "leggrid2d",
    "leggrid3d",
    "legvander2d",
    "legvander3d",
    "legcompanion",
    "leggauss",
    "legweight",
]

legtrim = polyutils.trimcoef


def poly2leg(pol):
    pol = polyutils.as_series(pol)
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
            c0 = polysub(c[i - 2], c1 * (i - 1) / i)
            c1 = polyadd(tmp, polymulx(c1, "same") * (2 * i - 1) / i)
            return c0, c1

        c0, c1 = jax.lax.fori_loop(0, n - 2, body, (c0, c1))

        return polyadd(c0, polymulx(c1, "same"))


legdomain = jax.numpy.array([-1, 1])


legzero = jax.numpy.array([0])


legone = jax.numpy.array([1])


legx = jax.numpy.array([0, 1])


def legline(off, scl):
    return jax.numpy.array([off, scl])


def legfromroots(roots):
    return polyutils._fromroots(legline, legmul, roots)


def legadd(c1, c2):
    return polyutils._add(c1, c2)


def legsub(c1, c2):
    return polyutils._sub(c1, c2)


def legmulx(c, mode="full"):
    c = polyutils.as_series(c)
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
    c1, c2 = polyutils.as_series(c1, c2)
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
    return polyutils._div(legmul, c1, c2)


def legpow(c, pow, maxpower=16):
    return polyutils._pow(legmul, c, pow, maxpower)


def legder(c, m=1, scl=1, axis=0):
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
    return polyutils._valnd(legval, c, x, y)


def leggrid2d(x, y, c):
    return polyutils._gridnd(legval, c, x, y)


def legval3d(x, y, z, c):
    return polyutils._valnd(legval, c, x, y, z)


def leggrid3d(x, y, z, c):
    return polyutils._gridnd(legval, c, x, y, z)


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
    return polyutils._vander_nd_flat((legvander, legvander), (x, y), deg)


def legvander3d(x, y, z, deg):
    return polyutils._vander_nd_flat((legvander, legvander, legvander), (x, y, z), deg)


def legfit(x, y, deg, rcond=None, full=False, w=None):
    return polyutils._fit(legvander, x, y, deg, rcond, full, w)


def legcompanion(c):
    c = polyutils.as_series(c)
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
    c = polyutils.as_series(c)
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
