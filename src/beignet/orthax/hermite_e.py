import jax
import jax.numpy

from . import polyutils

__all__ = [
    "hermezero",
    "hermeone",
    "hermex",
    "hermedomain",
    "hermeline",
    "hermeadd",
    "hermesub",
    "hermemulx",
    "hermemul",
    "hermediv",
    "hermepow",
    "hermeval",
    "hermeder",
    "hermeint",
    "herme2poly",
    "poly2herme",
    "hermefromroots",
    "hermevander",
    "hermefit",
    "hermetrim",
    "hermeroots",
    "hermeval2d",
    "hermeval3d",
    "hermegrid2d",
    "hermegrid3d",
    "hermevander2d",
    "hermevander3d",
    "hermecompanion",
    "hermegauss",
    "hermeweight",
]

hermetrim = polyutils.trimcoef


def poly2herme(pol):
    pol = polyutils.as_series(pol)
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

    c = polyutils.as_series(c)
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


hermedomain = jax.numpy.array([-1, 1])


hermezero = jax.numpy.array([0])


hermeone = jax.numpy.array([1])


hermex = jax.numpy.array([0, 1])


def hermeline(off, scl):
    return jax.numpy.array([off, scl])


def hermefromroots(roots):
    return polyutils._fromroots(hermeline, hermemul, roots)


def hermeadd(c1, c2):
    return polyutils._add(c1, c2)


def hermesub(c1, c2):
    return polyutils._sub(c1, c2)


def hermemulx(c, mode="full"):
    c = polyutils.as_series(c)
    prd = jax.numpy.zeros(len(c) + 1, dtype=c.dtype)
    prd = prd.at[1].set(c[0])

    i = jax.numpy.arange(1, len(c))

    prd = prd.at[i + 1].set(c[i])
    prd = prd.at[i - 1].add(c[i] * i)

    if mode == "same":
        prd = prd[: len(c)]
    return prd


def hermemul(c1, c2, mode="full"):
    c1, c2 = polyutils.as_series(c1, c2)
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
    return polyutils._div(hermemul, c1, c2)


def hermepow(c, pow, maxpower=16):
    return polyutils._pow(hermemul, c, pow, maxpower)


def hermeder(c, m=1, scl=1, axis=0):
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
            j = jax.numpy.arange(n, 0, -1)
            der = der.at[j - 1].set((j * c[j].T).T)
            c = der
    c = jax.numpy.moveaxis(c, 0, axis)
    return c


def hermeint(c, m=1, k=None, lbnd=0, scl=1, axis=0):
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
        j = jax.numpy.arange(1, n)
        tmp = tmp.at[j + 1].set((c[j].T / (j + 1)).T)
        tmp = tmp.at[0].add(k[i] - hermeval(lbnd, tmp))
        c = tmp
    c = jax.numpy.moveaxis(c, 0, axis)
    return c


def hermeval(x, c, tensor=True):
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
            c0 = c[-i] - c1 * (nd - 1)
            c1 = tmp + c1 * x
            return c0, c1, nd

        c0, c1, _ = jax.lax.fori_loop(3, len(c) + 1, body, (c0, c1, nd))

    return c0 + c1 * x


def hermeval2d(x, y, c):
    return polyutils._valnd(hermeval, c, x, y)


def hermegrid2d(x, y, c):
    return polyutils._gridnd(hermeval, c, x, y)


def hermeval3d(x, y, z, c):
    return polyutils._valnd(hermeval, c, x, y, z)


def hermegrid3d(x, y, z, c):
    return polyutils._gridnd(hermeval, c, x, y, z)


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
    return polyutils._vander_nd_flat((hermevander, hermevander), (x, y), deg)


def hermevander3d(x, y, z, deg):
    return polyutils._vander_nd_flat(
        (hermevander, hermevander, hermevander), (x, y, z), deg
    )


def hermefit(x, y, deg, rcond=None, full=False, w=None):
    return polyutils._fit(hermevander, x, y, deg, rcond, full, w)


def hermecompanion(c):
    c = polyutils.as_series(c)
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
    c = polyutils.as_series(c)
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
