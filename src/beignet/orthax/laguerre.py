import jax
import jax.numpy

from . import polyutils

__all__ = [
    "lagzero",
    "lagone",
    "lagx",
    "lagdomain",
    "lagline",
    "lagadd",
    "lagsub",
    "lagmulx",
    "lagmul",
    "lagdiv",
    "lagpow",
    "lagval",
    "lagder",
    "lagint",
    "lag2poly",
    "poly2lag",
    "lagfromroots",
    "lagvander",
    "lagfit",
    "lagtrim",
    "lagroots",
    "lagval2d",
    "lagval3d",
    "laggrid2d",
    "laggrid3d",
    "lagvander2d",
    "lagvander3d",
    "lagcompanion",
    "laggauss",
    "lagweight",
]

lagtrim = polyutils.trimcoef


def poly2lag(pol):
    pol = polyutils.as_series(pol)
    res = jax.numpy.zeros_like(pol)

    def body(i, res):
        res = lagadd(lagmulx(res, mode="same"), pol[::-1][i])
        return res

    res = jax.lax.fori_loop(0, len(pol), body, res)
    return res


def lag2poly(c):
    from .polynomial import polyadd, polymulx, polysub

    c = polyutils.as_series(c)
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


lagdomain = jax.numpy.array([0, 1])


lagzero = jax.numpy.array([0])


lagone = jax.numpy.array([1])


lagx = jax.numpy.array([1, -1])


def lagline(off, scl):
    return jax.numpy.array([off + scl, -scl])


def lagfromroots(roots):
    return polyutils._fromroots(lagline, lagmul, roots)


def lagadd(c1, c2):
    return polyutils._add(c1, c2)


def lagsub(c1, c2):
    return polyutils._sub(c1, c2)


def lagmulx(c, mode="full"):
    c = polyutils.as_series(c)

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
    c1, c2 = polyutils.as_series(c1, c2)
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
    return polyutils._div(lagmul, c1, c2)


def lagpow(c, pow, maxpower=16):
    return polyutils._pow(lagmul, c, pow, maxpower)


def lagder(c, m=1, scl=1, axis=0):
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
            c1 = tmp + (c1 * ((2 * nd - 1) - x)) / nd
            return c0, c1, nd

        c0, c1, _ = jax.lax.fori_loop(3, len(c) + 1, body, (c0, c1, nd))

    return c0 + c1 * (1 - x)


def lagval2d(x, y, c):
    return polyutils._valnd(lagval, c, x, y)


def laggrid2d(x, y, c):
    return polyutils._gridnd(lagval, c, x, y)


def lagval3d(x, y, z, c):
    return polyutils._valnd(lagval, c, x, y, z)


def laggrid3d(x, y, z, c):
    return polyutils._gridnd(lagval, c, x, y, z)


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
    return polyutils._vander_nd_flat((lagvander, lagvander), (x, y), deg)


def lagvander3d(x, y, z, deg):
    return polyutils._vander_nd_flat((lagvander, lagvander, lagvander), (x, y, z), deg)


def lagfit(x, y, deg, rcond=None, full=False, w=None):
    return polyutils._fit(lagvander, x, y, deg, rcond, full, w)


def lagcompanion(c):
    c = polyutils.as_series(c)
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
    c = polyutils.as_series(c)
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
