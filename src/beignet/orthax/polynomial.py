__all__ = [
    "polyzero",
    "polyone",
    "polyx",
    "polydomain",
    "polyline",
    "polyadd",
    "polysub",
    "polymulx",
    "polymul",
    "polydiv",
    "polypow",
    "polyval",
    "polyvalfromroots",
    "polyder",
    "polyint",
    "polyfromroots",
    "polyvander",
    "polyfit",
    "polytrim",
    "polyroots",
    "polyval2d",
    "polyval3d",
    "polygrid2d",
    "polygrid3d",
    "polyvander2d",
    "polyvander3d",
]


import jax
import jax.numpy

from . import polyutils

polytrim = polyutils.trimcoef

polydomain = jax.numpy.array([-1, 1])


polyzero = jax.numpy.array([0])


polyone = jax.numpy.array([1])


polyx = jax.numpy.array([0, 1])


def polyline(off, scl):
    return jax.numpy.array([off, scl])


def polyfromroots(roots):
    return polyutils._fromroots(polyline, polymul, roots)


def polyadd(c1, c2):
    return polyutils._add(c1, c2)


def polysub(c1, c2):
    return polyutils._sub(c1, c2)


def polymulx(c, mode="full"):
    c = polyutils.as_series(c)
    prd = jax.numpy.zeros(len(c) + 1, dtype=c.dtype)
    prd = prd.at[1:].set(c)
    if mode == "same":
        prd = prd[: len(c)]
    return prd


def polymul(c1, c2, mode="full"):
    c1, c2 = polyutils.as_series(c1, c2)
    ret = jax.numpy.convolve(c1, c2)
    if mode == "same":
        ret = ret[: max(len(c1), len(c2))]
    return ret


def polydiv(c1, c2):
    c1, c2 = polyutils.as_series(c1, c2)
    return polyutils._div(polymul, c1, c2)


def polypow(c, pow, maxpower=16):
    return polyutils._pow(polymul, c, pow, maxpower)


def polyder(c, m=1, scl=1, axis=0):
    c = polyutils.as_series(c)

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
    c = polyutils.as_series(c)
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
    c = polyutils._pad_along_axis(c, (0, m), axis)
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
    c = polyutils.as_series(c)
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
    return polyutils._valnd(polyval, c, x, y)


def polygrid2d(x, y, c):
    return polyutils._gridnd(polyval, c, x, y)


def polyval3d(x, y, z, c):
    return polyutils._valnd(polyval, c, x, y, z)


def polygrid3d(x, y, z, c):
    return polyutils._gridnd(polyval, c, x, y, z)


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
    return polyutils._vander_nd_flat((polyvander, polyvander), (x, y), deg)


def polyvander3d(x, y, z, deg):
    return polyutils._vander_nd_flat(
        (polyvander, polyvander, polyvander), (x, y, z), deg
    )


def polyfit(x, y, deg, rcond=None, full=False, w=None):
    return polyutils._fit(polyvander, x, y, deg, rcond, full, w)


def polycompanion(c):
    c = polyutils.as_series(c)
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
    c = polyutils.as_series(c)
    if len(c) < 2:
        return jax.numpy.array([], dtype=c.dtype)
    if len(c) == 2:
        return jax.numpy.array([-c[0] / c[1]])

    m = polycompanion(c)[::-1, ::-1]
    r = jax.numpy.linalg.eigvals(m)
    r = jax.numpy.sort(r)
    return r
