import functools
import operator

import jax
import jax.numpy
import numpy

__all__ = [
    "as_series",
    "trimseq",
    "trimcoef",
    "getdomain",
    "mapdomain",
    "mapparms",
]


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


@jax.jit
def getdomain(x):
    x = jax.numpy.asarray(x)
    if jax.numpy.iscomplexobj(x):
        rmin, rmax = x.real.min(), x.real.max()
        imin, imax = x.imag.min(), x.imag.max()
        return jax.numpy.array(((rmin + 1j * imin), (rmax + 1j * imax)))
    else:
        return jax.numpy.array((x.min(), x.max()))


@jax.jit
def mapparms(old, new):
    oldlen = old[1] - old[0]
    newlen = new[1] - new[0]
    off = (old[1] * new[0] - old[0] * new[1]) / oldlen
    scl = newlen / oldlen
    return off, scl


@jax.jit
def mapdomain(x, old, new):
    x = jax.numpy.asarray(x)
    off, scl = mapparms(old, new)
    return off + scl * x


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
