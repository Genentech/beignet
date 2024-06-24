import functools
import math
import operator
from typing import Callable, Literal, Union

import jax
import jax.numpy
from jax.numpy import (
    abs,
    arange,
    array,
    asarray,
    complexfloating,
    convolve,
    cos,
    cumprod,
    dot,
    empty,
    exp,
    finfo,
    full,
    hstack,
    iscomplexobj,
    iterable,
    linspace,
    moveaxis,
    ndim,
    newaxis,
    nonzero,
    ones,
    ones_like,
    prod,
    promote_types,
    roll,
    sin,
    sort,
    sqrt,
    square,
    stack,
    where,
    zeros,
    zeros_like,
)
from jax.numpy.linalg import (
    eigvals,
    eigvalsh,
    lstsq,
)
from jaxtyping import Array, Num

chebdomain = array([-1, 1])
chebone = array([1])
chebx = array([0, 1])
chebzero = array([0])
hermdomain = array([-1, 1])
hermedomain = array([-1, 1])
hermeone = array([1])
hermex = array([0, 1])
hermezero = array([0])
hermone = array([1])
hermx = array([0, 1 / 2])
hermzero = array([0])
lagdomain = array([0, 1])
lagone = array([1])
lagx = array([1, -1])
lagzero = array([0])
legdomain = array([-1, 1])
legone = array([1])
legx = array([0, 1])
legzero = array([0])
polydomain = array([-1, 1])
polyone = array([1])
polyx = array([0, 1])
polyzero = array([0])


def scan(f, init, xs, length=None):
    if xs is None:
        xs = [None] * length
    carry = init
    ys = []
    for x in xs:
        carry, y = f(carry, x)
        ys.append(y)
    return carry, stack(ys)


def _add(
    input: Num[Array, "..."],
    other: Num[Array, "..."],
) -> Num[Array, "..."]:
    input, other = _as_series(input, other)

    if len(input) > len(other):
        return input.at[: other.size].add(other)

    return other.at[: input.size].add(input)


def _c_series_to_z_series(
    input: Num[Array, "..."],
) -> Num[Array, "..."]:
    n = input.size

    zs = zeros(2 * n - 1, dtype=input.dtype)

    zs = zs.at[n - 1 :].set(input / 2)

    return zs + zs[::-1]


def _div(
    func,
    input: Num[Array, "..."],
    other: Num[Array, "..."],
) -> Union[Num[Array, "..."], Num[Array, "..."]]:
    input, other = _as_series(input, other)

    lc1 = len(input)
    lc2 = len(other)

    if lc1 < lc2:
        return zeros_like(input[:1]), input

    if lc2 == 1:
        return input / other[-1], zeros_like(input[:1])

    def _ldordidx(x):  # index of highest order nonzero term
        return len(x) - 1 - nonzero(x[::-1], size=1)[0][0]

    quo = zeros(lc1 - lc2 + 1, dtype=input.dtype)
    rem = input
    ridx = len(rem) - 1
    sz = lc1 - _ldordidx(other) - 1
    y = zeros(lc1 + lc2 + 1, dtype=input.dtype).at[sz].set(1.0)

    def body(k, val):
        quo, rem, y, ridx = val
        i = sz - k
        p = func(y, other)
        pidx = _ldordidx(p)
        t = rem[ridx] / p[pidx]
        rem = _subtract(rem.at[ridx].set(0), t * p.at[pidx].set(0))[: len(rem)]
        quo = quo.at[i].set(t)
        ridx -= 1
        y = roll(y, -1)
        return quo, rem, y, ridx

    x = (quo, rem, y, ridx)
    y1 = x
    for index in range(0, sz):
        y1 = body(index, y1)
    quo, rem, _, _ = y1
    return quo, rem


def _fit(vander_f, x, y, degree, rcond=None, full=False, w=None):  # noqa:C901
    x = asarray(x)
    y = asarray(y)
    degree = asarray(degree)

    if degree.ndim > 1 or degree.dtype.kind not in "iu" or degree.size == 0:
        raise TypeError

    if degree.min() < 0:
        raise ValueError

    if x.ndim != 1:
        raise TypeError

    if x.size == 0:
        raise TypeError

    if y.ndim < 1 or y.ndim > 2:
        raise TypeError

    if len(x) != len(y):
        raise TypeError

    if degree.ndim == 0:
        lmax = int(degree)
        van = vander_f(x, lmax)
    else:
        degree = sort(degree)
        lmax = int(degree[-1])
        van = vander_f(x, lmax)[:, degree]

    lhs = van.T
    rhs = y.T

    if w is not None:
        w = asarray(w)

        if w.ndim != 1:
            raise TypeError

        if len(x) != len(w):
            raise TypeError

        lhs = lhs * w
        rhs = rhs * w

    if rcond is None:
        rcond = len(x) * finfo(x.dtype).eps

    if issubclass(lhs.dtype.type, complexfloating):
        scl = sqrt((square(lhs.real) + square(lhs.imag)).sum(1))
    else:
        scl = sqrt(square(lhs).sum(1))
    scl = where(scl == 0, 1, scl)

    c, resids, rank, s = lstsq(lhs.T / scl, rhs.T, rcond)
    c = (c.T / scl).T

    if degree.ndim > 0:
        if c.ndim == 2:
            cc = zeros((lmax + 1, c.shape[1]), dtype=c.dtype)
        else:
            cc = zeros(lmax + 1, dtype=c.dtype)
        cc = cc.at[degree].set(c)
        c = cc

    if full:
        return c, [resids, rank, s, rcond]
    else:
        return c


def _from_roots(f, g, input):
    input = asarray(input)

    if input.size == 0:
        return ones(1)

    input = sort(input)

    retlen = len(input) + 1

    def p_scan_fun(carry, x):
        return carry, _add(zeros(retlen, dtype=x.dtype), f(-x, 1))

    _, p = scan(p_scan_fun, 0, input)

    p = asarray(p)
    n = len(p)

    def body_fun(val):
        m, r = divmod(val[0], 2)
        arr = val[1]
        tmp = array([zeros(retlen, dtype=p.dtype)] * len(p))

        def inner_body_fun(i, val):
            return val.at[i].set(g(arr[i], arr[i + m])[:retlen])

        tmp = jax.lax.fori_loop(0, m, inner_body_fun, tmp)

        if r:
            tmp = tmp.at[0].set(g(tmp[0], arr[2 * m])[:retlen])

        return m, tmp

    val = (n, p)

    while val[0] > 1:
        val = body_fun(val)

    _, ret = val

    return ret[0]


def _gridnd(val_f, c, *args):
    for xi in args:
        c = val_f(xi, c)

    return c


def _normed_hermite_e_n(x, n):
    def truefun():
        return full(x.shape, 1 / sqrt(sqrt(2 * math.pi)))

    def falsefun():
        c0 = zeros_like(x)
        c1 = ones_like(x) / sqrt(sqrt(2 * math.pi))
        nd = array(n).astype(float)

        def body(i, val):
            c0, c1, nd = val
            tmp = c0
            c0 = -c1 * sqrt((nd - 1.0) / nd)
            c1 = tmp + c1 * x * sqrt(1.0 / nd)
            nd = nd - 1.0
            return c0, c1, nd

        b = n - 1

        x1 = (c0, c1, nd)

        y = x1

        for index in range(0, b):
            y = body(index, y)

        c0, c1, _ = y

        return c0 + c1 * x

    if n == 0:
        output = truefun()
    else:
        output = falsefun()

    return output


def _normed_hermite_n(x, n):
    def truefun():
        return full(x.shape, 1 / sqrt(sqrt(math.pi)))

    def falsefun():
        c0 = zeros_like(x)
        c1 = ones_like(x) / sqrt(sqrt(math.pi))
        nd = array(n).astype(float)

        def body(i, val):
            c0, c1, nd = val
            tmp = c0
            c0 = -c1 * sqrt((nd - 1.0) / nd)
            c1 = tmp + c1 * x * sqrt(2.0 / nd)
            nd = nd - 1.0
            return c0, c1, nd

        b = n - 1
        x1 = (c0, c1, nd)
        y = x1
        for index in range(0, b):
            y = body(index, y)
        c0, c1, _ = y
        return c0 + c1 * x * sqrt(2)

    if n == 0:
        output = truefun()
    else:
        output = falsefun()

    return output


def _nth_slice(i, ndim):
    sl = [newaxis] * ndim
    sl[i] = slice(None)
    return tuple(sl)


def _pad_along_axis(input, pad=(0, 0), axis=0):
    input = moveaxis(input, axis, 0)

    if pad[0] < 0:
        input = input[abs(pad[0]) :]
        pad = (0, pad[1])
    if pad[1] < 0:
        input = input[: -abs(pad[1])]
        pad = (pad[0], 0)

    npad = [(0, 0)] * input.ndim
    npad[0] = pad

    output = jax.numpy.pad(input, pad_width=npad, mode="constant", constant_values=0)
    return moveaxis(output, 0, axis)


def _pow(
    func: Callable[
        [
            Num[Array, "..."],
            Num[Array, "..."],
            Literal["full", "same"],
        ],
        Num[Array, "..."],
    ],
    input: Num[Array, "..."],
    exponent,
    maximum_exponent,
):
    input = _as_series(input)

    power = int(exponent)

    if power != exponent or power < 0:
        raise ValueError

    if maximum_exponent is not None and power > maximum_exponent:
        raise ValueError

    if power == 0:
        return array([1], dtype=input.dtype)

    if power == 1:
        return input

    output = zeros(len(input) * exponent, dtype=input.dtype)

    output = _add(output, input)

    for _ in range(2, power + 1):
        output = func(output, input, mode="same")

    return output


def _subtract(
    input: Num[Array, "..."],
    other: Num[Array, "..."],
) -> Num[Array, "..."]:
    input, other = _as_series(input, other)

    if len(input) > len(other):
        output = -other

        output = input.at[: other.size].add(output)

        return output

    output = -other

    output = output.at[: input.size].add(input)

    return output


def _evaluate(
    func: Callable,
    input: Num[Array, "..."],
    *args,
):
    xs = []

    for a in args:
        xs = [*xs, asarray(a)]

    if not all(a.shape == xs[0].shape for a in xs[1:]):
        match len(xs):
            case 2:
                raise ValueError
            case 3:
                raise ValueError
            case _:
                raise ValueError

    xs = iter(xs)

    output = func(next(xs), input)

    for x in xs:
        output = func(x, output, tensor=False)

    return output


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

    points = tuple(array(tuple(points), copy=False) + 0.0)

    vander_arrays = (
        vander_fs[i](points[i], degrees[i])[(...,) + _nth_slice(i, n_dims)]
        for i in range(n_dims)
    )

    return functools.reduce(operator.mul, vander_arrays)


def _vander_nd_flat(vander_fs, points, degrees):
    v = _vander_nd(vander_fs, points, degrees)
    return v.reshape(v.shape[: -len(degrees)] + (-1,))


def _z_series_mul(z1, z2, mode="full"):
    return convolve(z1, z2, mode=mode)


def _z_series_to_c_series(zs):
    n = (zs.size + 1) // 2
    c = zs[n - 1 :].copy()
    return c.at[1:n].multiply(2)


def _as_series(*arrs, trim=False):
    arrays = tuple(array(a, ndmin=1) for a in arrs)

    if trim:
        arrays = tuple(_trim_sequence(a) for a in arrays)

    arrays = jax._src.numpy.util.promote_dtypes_inexact(*arrays)

    if len(arrays) == 1:
        return arrays[0]

    return tuple(arrays)


def cheb2poly(c):
    c = _as_series(c)

    n = len(c)

    if n < 3:
        return c

    c0 = zeros_like(c).at[0].set(c[-2])
    c1 = zeros_like(c).at[0].set(c[-1])

    def body(k, c0c1):
        i = n - 1 - k

        c0, c1 = c0c1

        tmp = c0

        c0 = polysub(c[i - 2], c1)

        c1 = polyadd(tmp, polymulx(c1, "same") * 2)

        return c0, c1

    b = n - 2

    x = (c0, c1)

    y = x

    for index in range(0, b):
        y = body(index, y)

    c0, c1 = y

    return polyadd(c0, polymulx(c1, "same"))


def chebadd(
    input: Num[Array, "..."],
    other: Num[Array, "..."],
) -> Num[Array, "..."]:
    return _add(input, other)


def chebcompanion(c):
    c = _as_series(c)
    if len(c) < 2:
        raise ValueError("Series must have maximum degree of at least 1.")
    if len(c) == 2:
        return array([[-c[0] / c[1]]])

    n = len(c) - 1
    mat = zeros((n, n), dtype=c.dtype)
    scl = ones(n).at[1:].set(sqrt(0.5))
    shp = mat.shape
    mat = mat.flatten()
    mat = mat.at[1 :: n + 1].set(full(n - 1, 1 / 2).at[0].set(sqrt(0.5)))
    mat = mat.at[n :: n + 1].set(full(n - 1, 1 / 2).at[0].set(sqrt(0.5)))
    mat = mat.reshape(shp)
    mat = mat.at[:, -1].add(-(c[:-1] / c[-1]) * (scl / scl[-1]) * 0.5)
    return mat


def chebder(c, m=1, scl=1, axis=0):
    if m < 0:
        raise ValueError("The order of derivation must be non-negative")

    c = _as_series(c)

    if m == 0:
        return c

    c = moveaxis(c, axis, 0)
    n = len(c)
    if m >= n:
        c = zeros_like(c[:1])
    else:
        for _ in range(m):
            n = n - 1
            c *= scl
            der = empty((n,) + c.shape[1:], dtype=c.dtype)

            def body(k, der_c, n=n):
                j = n - k
                der, c = der_c
                der = der.at[j - 1].set((2 * j) * c[j])
                c = c.at[j - 2].add((j * c[j]) / (j - 2))
                return der, c

            b = n - 2
            x = (der, c)
            y = x
            for index in range(0, b):
                y = body(index, y)
            der, c = y

            if n > 1:
                der = der.at[1].set(4 * c[2])
            der = der.at[0].set(c[1])
            c = der

    c = moveaxis(c, 0, axis)
    return c


def chebdiv(
    input: Num[Array, "..."],
    other: Num[Array, "..."],
) -> (Num[Array, "..."], Num[Array, "..."]):
    return _div(chebmul, input, other)


def chebfit(x, y, degree, rcond=None, full=False, w=None):
    return _fit(chebvander, x, y, degree, rcond, full, w)


def chebfromroots(roots):
    return _from_roots(chebline, chebmul, roots)


def chebgauss(degree):
    degree = int(degree)

    if degree <= 0:
        raise ValueError

    output = arange(1, 2 * degree, 2)

    output = output / (2.0 * degree)

    output = output * math.pi

    output = cos(output)

    w = ones(degree) * (math.pi / degree)

    return output, w


def chebgrid2d(x, y, c):
    return _gridnd(chebval, c, x, y)


def chebgrid3d(x, y, z, c):
    return _gridnd(chebval, c, x, y, z)


def chebint(c, m=1, k=None, lbnd=0, scl=1, axis=0):
    if k is None:
        k = []
    c = _as_series(c)
    lbnd, scl = map(asarray, (lbnd, scl))

    if not iterable(k):
        k = [k]
    if len(k) > m:
        raise ValueError("Too many integration constants")
    if ndim(lbnd) != 0:
        raise ValueError("lbnd must be a scalar.")
    if ndim(scl) != 0:
        raise ValueError("scl must be a scalar.")

    if m == 0:
        return c

    c = moveaxis(c, axis, 0)
    k = array(list(k) + [0] * (m - len(k)), ndmin=1)

    for i in range(m):
        n = len(c)
        c *= scl
        tmp = empty((n + 1,) + c.shape[1:], dtype=c.dtype)
        tmp = tmp.at[0].set(c[0] * 0)
        tmp = tmp.at[1].set(c[0])
        if n > 1:
            tmp = tmp.at[2].set(c[1] / 4)
        j = arange(2, n)
        tmp = tmp.at[j + 1].set((c[j].T / (2 * (j + 1))).T)
        tmp = tmp.at[j - 1].add(-(c[j].T / (2 * (j - 1))).T)
        tmp = tmp.at[0].add(k[i] - chebval(lbnd, tmp))
        c = tmp
    c = moveaxis(c, 0, axis)
    return c


def chebinterpolate(func, degree, args=()):
    _deg = int(degree)
    if _deg != degree:
        raise ValueError("degree must be integer")
    if _deg < 0:
        raise ValueError("expected degree >= 0")

    order = _deg + 1
    xcheb = chebpts1(order)
    yfunc = func(xcheb, *args)
    m = chebvander(xcheb, _deg)
    c = dot(m.T, yfunc)
    c = c.at[0].divide(order)
    c = c.at[1:].divide(0.5 * order)

    return c


def chebline(off, scl):
    return array([off, scl])


def chebmul(
    input: Num[Array, "..."],
    other: Num[Array, "..."],
    mode: Literal["full", "same"] = "full",
) -> Num[Array, "..."]:
    input, other = _as_series(input, other)

    z1 = _c_series_to_z_series(input)
    z2 = _c_series_to_z_series(other)

    output = _z_series_mul(z1, z2, mode=mode)

    output = _z_series_to_c_series(output)

    if mode == "same":
        output = output[: max(len(input), len(other))]

    return output


def chebmulx(c, mode="full"):
    c = _as_series(c)

    output = zeros(len(c) + 1, dtype=c.dtype)

    output = output.at[1].set(c[0])

    if len(c) > 1:
        tmp = c[1:] / 2

        output = output.at[2:].set(tmp)

        output = output.at[0:-2].add(tmp)

    if mode == "same":
        output = output[: len(c)]

    return output


def chebpow(c, pow, maxpower=16):
    c = _as_series(c)

    power = int(pow)

    if power != pow or power < 0:
        raise ValueError

    if maxpower is not None and power > maxpower:
        raise ValueError

    if power == 0:
        return array([1], dtype=c.dtype)

    if power == 1:
        return c

    output = zeros(len(c) * pow, dtype=c.dtype)

    output = chebadd(output, c)

    zs = _c_series_to_z_series(c)

    output = _c_series_to_z_series(output)

    def func(_, p):
        return convolve(p, zs, mode="same")

    b = power + 1
    y = output
    for index in range(2, b):
        y = func(index, y)
    output = y

    output = _z_series_to_c_series(output)

    return output


def chebpts1(points):
    _points = int(points)

    if _points != points:
        raise ValueError

    if _points < 1:
        raise ValueError

    output = arange(-_points + 1, _points + 1, 2)

    output = 0.5 * math.pi / _points * output

    output = sin(output)

    return output


def chebpts2(points):
    _points = int(points)

    if _points != points:
        raise ValueError

    if _points < 2:
        raise ValueError

    output = linspace(-math.pi, 0, _points)

    output = cos(output)

    return output


def chebroots(c):
    c = _as_series(c)

    if len(c) <= 1:
        return array([], dtype=c.dtype)

    if len(c) == 2:
        return array([-c[0] / c[1]])

    output = chebcompanion(c)

    output = output[::-1, ::-1]

    output = eigvals(output)

    output = sort(output)

    return output


def chebsub(
    input: Num[Array, "..."],
    other: Num[Array, "..."],
) -> Num[Array, "..."]:
    return _subtract(input, other)


def chebval(x, c, tensor=True):
    c = _as_series(c)
    x = asarray(x)
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
        c0 = c[-2] * ones_like(x)
        c1 = c[-1] * ones_like(x)

        def body(i, val):
            c0, c1 = val
            tmp = c0
            c0 = c[-i] - c1
            c1 = tmp + c1 * x2
            return c0, c1

        b = len(c) + 1
        x1 = (c0, c1)
        y = x1
        for index in range(3, b):
            y = body(index, y)
        c0, c1 = y

    return c0 + c1 * x


def chebval2d(x, y, c):
    return _evaluate(chebval, c, x, y)


def chebval3d(x, y, z, c):
    return _evaluate(chebval, c, x, y, z)


def chebvander(x, degree):
    if degree < 0:
        raise ValueError

    x = array(x, ndmin=1)
    dims = (degree + 1,) + x.shape
    dtyp = promote_types(x.dtype, array(0.0).dtype)
    x = x.astype(dtyp)
    v = empty(dims, dtype=dtyp)
    v = v.at[0].set(ones_like(x))

    if degree > 0:
        v = v.at[1].set(x)
        x2 = 2 * x

        def body(i, v):
            return v.at[i].set(v[i - 1] * x2 - v[i - 2])

        b = degree + 1
        y = v
        for index in range(2, b):
            y = body(index, y)
        v = y

    return moveaxis(v, 0, -1)


def chebvander2d(x, y, degree):
    return _vander_nd_flat(
        (chebvander, chebvander),
        (x, y),
        degree,
    )


def chebvander3d(x, y, z, degree):
    return _vander_nd_flat(
        (chebvander, chebvander, chebvander),
        (x, y, z),
        degree,
    )


def chebweight(x):
    x = asarray(x)

    output = sqrt(1.0 + x) * sqrt(1.0 - x)

    output = 1.0 / output

    return output


def getdomain(x):
    x = asarray(x)

    if iscomplexobj(x):
        rmin, rmax = x.real.min(), x.real.max()
        imin, imax = x.imag.min(), x.imag.max()
        return array(((rmin + 1j * imin), (rmax + 1j * imax)))

    return array((x.min(), x.max()))


def herm2poly(c):
    c = _as_series(c)
    n = len(c)
    if n == 1:
        return c
    if n == 2:
        c = c.at[1].multiply(2)
        return c
    else:
        c0 = zeros_like(c).at[0].set(c[-2])
        c1 = zeros_like(c).at[0].set(c[-1])

        def body(k, c0c1):
            i = n - 1 - k
            c0, c1 = c0c1
            tmp = c0
            c0 = polysub(c[i - 2], c1 * (2 * (i - 1)))
            c1 = polyadd(tmp, polymulx(c1, "same") * 2)
            return c0, c1

        b = n - 2
        x = (c0, c1)
        y = x
        for index in range(0, b):
            y = body(index, y)
        c0, c1 = y

        return polyadd(c0, polymulx(c1, "same") * 2)


def hermadd(
    input: Num[Array, "..."],
    other: Num[Array, "..."],
) -> Num[Array, "..."]:
    return _add(input, other)


def hermcompanion(c):
    c = _as_series(c)
    if len(c) < 2:
        raise ValueError("Series must have maximum degree of at least 1.")
    if len(c) == 2:
        return array([[-0.5 * c[0] / c[1]]])

    n = len(c) - 1
    mat = zeros((n, n), dtype=c.dtype)
    scl = hstack((1.0, 1.0 / sqrt(2.0 * arange(n - 1, 0, -1))))
    scl = cumprod(scl)[::-1]
    shp = mat.shape
    mat = mat.flatten()
    mat = mat.at[1 :: n + 1].set(sqrt(0.5 * arange(1, n)))
    mat = mat.at[n :: n + 1].set(sqrt(0.5 * arange(1, n)))
    mat = mat.reshape(shp)
    mat = mat.at[:, -1].add(-scl * c[:-1] / (2.0 * c[-1]))
    return mat


def hermder(c, m=1, scl=1, axis=0):
    if m < 0:
        raise ValueError("The order of derivation must be non-negative")

    c = _as_series(c)

    if m == 0:
        return c

    c = moveaxis(c, axis, 0)
    n = len(c)
    if m >= n:
        c = zeros_like(c[:1])
    else:
        for _ in range(m):
            n = n - 1
            c *= scl
            der = empty((n,) + c.shape[1:], dtype=c.dtype)
            j = arange(n, 0, -1)
            der = der.at[j - 1].set((2 * j * c[j].T).T)
            c = der
    c = moveaxis(c, 0, axis)
    return c


def hermdiv(
    input: Num[Array, "..."],
    other: Num[Array, "..."],
) -> (Num[Array, "..."], Num[Array, "..."]):
    return _div(hermmul, input, other)


def herme2poly(c):
    c = _as_series(c)
    n = len(c)
    if n == 1:
        return c
    if n == 2:
        return c
    else:
        c0 = zeros_like(c).at[0].set(c[-2])
        c1 = zeros_like(c).at[0].set(c[-1])

        def body(k, c0c1):
            i = n - 1 - k
            c0, c1 = c0c1
            tmp = c0
            c0 = polysub(c[i - 2], c1 * (i - 1))
            c1 = polyadd(tmp, polymulx(c1, "same"))
            return c0, c1

        b = n - 2
        x = (c0, c1)
        y = x
        for index in range(0, b):
            y = body(index, y)
        c0, c1 = y

        return polyadd(c0, polymulx(c1, "same"))


def hermeadd(
    input: Num[Array, "..."],
    other: Num[Array, "..."],
) -> Num[Array, "..."]:
    return _add(input, other)


def hermecompanion(c):
    c = _as_series(c)
    if len(c) < 2:
        raise ValueError("Series must have maximum degree of at least 1.")
    if len(c) == 2:
        return array([[-c[0] / c[1]]])

    n = len(c) - 1
    mat = zeros((n, n), dtype=c.dtype)
    scl = hstack((1.0, 1.0 / sqrt(arange(n - 1, 0, -1))))
    scl = cumprod(scl)[::-1]
    shp = mat.shape
    mat = mat.flatten()
    mat = mat.at[1 :: n + 1].set(sqrt(arange(1, n)))
    mat = mat.at[n :: n + 1].set(sqrt(arange(1, n)))
    mat = mat.reshape(shp)
    mat = mat.at[:, -1].add(-scl * c[:-1] / c[-1])
    return mat


def hermeder(c, m=1, scl=1, axis=0):
    if m < 0:
        raise ValueError("The order of derivation must be non-negative")

    c = _as_series(c)

    if m == 0:
        return c

    c = moveaxis(c, axis, 0)
    n = len(c)
    if m >= n:
        c = zeros_like(c[:1])
    else:
        for _ in range(m):
            n = n - 1
            c *= scl
            der = empty((n,) + c.shape[1:], dtype=c.dtype)
            j = arange(n, 0, -1)
            der = der.at[j - 1].set((j * c[j].T).T)
            c = der
    c = moveaxis(c, 0, axis)
    return c


def hermediv(
    input: Num[Array, "..."],
    other: Num[Array, "..."],
) -> (Num[Array, "..."], Num[Array, "..."]):
    return _div(hermemul, input, other)


def hermefit(x, y, degree, rcond=None, full=False, w=None):
    return _fit(hermevander, x, y, degree, rcond, full, w)


def hermefromroots(roots):
    return _from_roots(hermeline, hermemul, roots)


def hermegauss(degree):
    degree = int(degree)
    if degree <= 0:
        raise ValueError("degree must be a positive integer")

    c = zeros(degree + 1).at[-1].set(1)
    m = hermecompanion(c)
    x = eigvalsh(m)

    dy = _normed_hermite_e_n(x, degree)
    df = _normed_hermite_e_n(x, degree - 1) * sqrt(degree)
    x -= dy / df

    fm = _normed_hermite_e_n(x, degree - 1)
    fm /= abs(fm).max()
    w = 1 / (fm * fm)

    w = (w + w[::-1]) / 2
    x = (x - x[::-1]) / 2

    w *= sqrt(2 * math.pi) / w.sum()

    return x, w


def hermegrid2d(x, y, c):
    return _gridnd(hermeval, c, x, y)


def hermegrid3d(x, y, z, c):
    return _gridnd(hermeval, c, x, y, z)


def hermeint(c, m=1, k=None, lbnd=0, scl=1, axis=0):
    if k is None:
        k = []

    c = _as_series(c)

    lbnd, scl = map(asarray, (lbnd, scl))

    if not iterable(k):
        k = [k]

    if len(k) > m:
        raise ValueError("Too many integration constants")

    if ndim(lbnd) != 0:
        raise ValueError("lbnd must be a scalar.")

    if ndim(scl) != 0:
        raise ValueError("scl must be a scalar.")

    if m == 0:
        return c

    c = moveaxis(c, axis, 0)
    k = array(list(k) + [0] * (m - len(k)), ndmin=1)

    for i in range(m):
        n = len(c)
        c *= scl
        tmp = empty((n + 1,) + c.shape[1:], dtype=c.dtype)
        tmp = tmp.at[0].set(c[0] * 0)
        tmp = tmp.at[1].set(c[0])
        j = arange(1, n)
        tmp = tmp.at[j + 1].set((c[j].T / (j + 1)).T)
        tmp = tmp.at[0].add(k[i] - hermeval(lbnd, tmp))
        c = tmp

    return moveaxis(c, 0, axis)


def hermeline(off, scl):
    return array([off, scl])


def hermemul(input, other, mode="full"):
    input, other = _as_series(input, other)
    lc1, lc2 = len(input), len(other)
    if lc1 > lc2:
        c = other
        xs = input
    else:
        c = input
        xs = other

    if len(c) == 1:
        c0 = hermeadd(zeros(lc1 + lc2 - 1), c[0] * xs)
        input = zeros(lc1 + lc2 - 1)
    elif len(c) == 2:
        c0 = hermeadd(zeros(lc1 + lc2 - 1), c[0] * xs)
        input = hermeadd(zeros(lc1 + lc2 - 1), c[1] * xs)
    else:
        nd = len(c)
        c0 = hermeadd(zeros(lc1 + lc2 - 1), c[-2] * xs)
        input = hermeadd(zeros(lc1 + lc2 - 1), c[-1] * xs)

        def body(i, val):
            c0, c1, nd = val
            tmp = c0
            nd = nd - 1
            c0 = hermesub(c[-i] * xs, c1 * (nd - 1))
            c1 = hermeadd(tmp, hermemulx(c1, "same"))
            return c0, c1, nd

        b = len(c) + 1
        x = (c0, input, nd)
        y = x
        for index in range(3, b):
            y = body(index, y)
        c0, input, _ = y

    ret = hermeadd(c0, hermemulx(input, "same"))
    if mode == "same":
        ret = ret[: max(lc1, lc2)]
    return ret


def hermemulx(c, mode="full"):
    c = _as_series(c)
    prd = zeros(len(c) + 1, dtype=c.dtype)
    prd = prd.at[1].set(c[0])

    i = arange(1, len(c))

    prd = prd.at[i + 1].set(c[i])
    prd = prd.at[i - 1].add(c[i] * i)

    if mode == "same":
        prd = prd[: len(c)]
    return prd


def hermepow(c, pow, maxpower=16):
    return _pow(hermemul, c, pow, maxpower)


def hermeroots(c):
    c = _as_series(c)

    if len(c) <= 1:
        return array([], dtype=c.dtype)

    if len(c) == 2:
        return array([-c[0] / c[1]])

    return sort(eigvals(hermecompanion(c)[::-1, ::-1]))


def hermesub(
    input: Num[Array, "..."],
    other: Num[Array, "..."],
) -> Num[Array, "..."]:
    return _subtract(input, other)


def hermeval(x, c, tensor=True):
    c = _as_series(c)
    x = asarray(x)
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
        c0 = c[-2] * ones_like(x)
        c1 = c[-1] * ones_like(x)

        def body(i, val):
            c0, c1, nd = val
            tmp = c0
            nd = nd - 1
            c0 = c[-i] - c1 * (nd - 1)
            c1 = tmp + c1 * x
            return c0, c1, nd

        b = len(c) + 1
        x1 = (c0, c1, nd)
        y = x1
        for index in range(3, b):
            y = body(index, y)
        c0, c1, _ = y

    return c0 + c1 * x


def hermeval2d(x, y, c):
    return _evaluate(hermeval, c, x, y)


def hermeval3d(x, y, z, c):
    return _evaluate(hermeval, c, x, y, z)


def hermevander(x, degree):
    if degree < 0:
        raise ValueError("degree must be non-negative")

    x = array(x, ndmin=1)
    dims = (degree + 1,) + x.shape
    dtyp = promote_types(x.dtype, array(0.0).dtype)
    x = x.astype(dtyp)
    v = empty(dims, dtype=dtyp)
    v = v.at[0].set(ones_like(x))
    if degree > 0:
        v = v.at[1].set(x)

        def body(i, v):
            return v.at[i].set(v[i - 1] * x - v[i - 2] * (i - 1))

        b = degree + 1
        y = v
        for index in range(2, b):
            y = body(index, y)
        v = y

    return moveaxis(v, 0, -1)


def hermevander2d(x, y, degree):
    return _vander_nd_flat(
        (hermevander, hermevander),
        (x, y),
        degree,
    )


def hermevander3d(x, y, z, degree):
    return _vander_nd_flat(
        (hermevander, hermevander, hermevander),
        (x, y, z),
        degree,
    )


def hermeweight(x):
    return exp(-0.5 * x**2)


def hermfit(x, y, degree, rcond=None, full=False, w=None):
    return _fit(hermvander, x, y, degree, rcond, full, w)


def hermfromroots(roots):
    return _from_roots(hermline, hermmul, roots)


def hermgauss(degree):
    degree = int(degree)
    if degree <= 0:
        raise ValueError("degree must be a positive integer")

    c = zeros(degree + 1).at[-1].set(1)
    x = eigvalsh(hermcompanion(c))

    dy = _normed_hermite_n(x, degree)
    df = _normed_hermite_n(x, degree - 1) * sqrt(2 * degree)
    x -= dy / df

    fm = _normed_hermite_n(x, degree - 1)
    fm /= abs(fm).max()
    w = 1 / (fm * fm)

    w = (w + w[::-1]) / 2
    x = (x - x[::-1]) / 2

    w = w * (sqrt(math.pi) / w.sum())

    return x, w


def hermgrid2d(x, y, c):
    return _gridnd(hermval, c, x, y)


def hermgrid3d(x, y, z, c):
    return _gridnd(hermval, c, x, y, z)


def hermint(c, m=1, k=None, lbnd=0, scl=1, axis=0):
    if k is None:
        k = []
    c = _as_series(c)
    lbnd, scl = map(asarray, (lbnd, scl))

    if not iterable(k):
        k = [k]

    if len(k) > m:
        raise ValueError("Too many integration constants")

    if ndim(lbnd) != 0:
        raise ValueError("lbnd must be a scalar.")

    if ndim(scl) != 0:
        raise ValueError("scl must be a scalar.")

    if m == 0:
        return c

    c = moveaxis(c, axis, 0)
    k = array(list(k) + [0] * (m - len(k)), ndmin=1)

    for i in range(m):
        n = len(c)
        c *= scl
        tmp = empty((n + 1,) + c.shape[1:], dtype=c.dtype)
        tmp = tmp.at[0].set(c[0] * 0)
        tmp = tmp.at[1].set(c[0] / 2)
        j = arange(1, n)
        tmp = tmp.at[j + 1].set((c[j].T / (2 * (j + 1))).T)
        tmp = tmp.at[0].add(k[i] - hermval(lbnd, tmp))
        c = tmp

    c = moveaxis(c, 0, axis)

    return c


def hermline(off, scl):
    return array([off, scl / 2])


def hermmul(input, other, mode="full"):
    input, other = _as_series(input, other)
    lc1, lc2 = len(input), len(other)
    if lc1 > lc2:
        c = other
        xs = input
    else:
        c = input
        xs = other

    if len(c) == 1:
        c0 = hermadd(zeros(lc1 + lc2 - 1), c[0] * xs)
        input = zeros(lc1 + lc2 - 1)
    elif len(c) == 2:
        c0 = hermadd(zeros(lc1 + lc2 - 1), c[0] * xs)
        input = hermadd(zeros(lc1 + lc2 - 1), c[1] * xs)
    else:
        nd = len(c)
        c0 = hermadd(zeros(lc1 + lc2 - 1), c[-2] * xs)
        input = hermadd(zeros(lc1 + lc2 - 1), c[-1] * xs)

        def body(i, val):
            c0, c1, nd = val
            tmp = c0
            nd = nd - 1
            c0 = hermsub(c[-i] * xs, c1 * (2 * (nd - 1)))
            c1 = hermadd(tmp, hermmulx(c1, "same") * 2)
            return c0, c1, nd

        b = len(c) + 1
        x = (c0, input, nd)
        y = x
        for index in range(3, b):
            y = body(index, y)
        c0, input, _ = y

    ret = hermadd(c0, hermmulx(input, "same") * 2)
    if mode == "same":
        ret = ret[: max(lc1, lc2)]
    return ret


def hermmulx(c, mode="full"):
    c = _as_series(c)
    prd = zeros(len(c) + 1, dtype=c.dtype)
    prd = prd.at[1].set(c[0] / 2)

    i = arange(1, len(c))

    prd = prd.at[i + 1].set(c[i] / 2)
    prd = prd.at[i - 1].add(c[i] * i)

    if mode == "same":
        prd = prd[: len(c)]

    return prd


def hermpow(c, pow, maxpower=16):
    return _pow(hermmul, c, pow, maxpower)


def hermroots(c):
    c = _as_series(c)

    if len(c) <= 1:
        return array([], dtype=c.dtype)

    if len(c) == 2:
        return array([-0.5 * c[0] / c[1]])

    return sort(eigvals(hermcompanion(c)[::-1, ::-1]))


def hermsub(
    input: Num[Array, "..."],
    other: Num[Array, "..."],
) -> Num[Array, "..."]:
    return _subtract(input, other)


def hermval(x, c, tensor=True):
    c = _as_series(c)
    x = asarray(x)
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
        c0 = c[-2] * ones_like(x)
        c1 = c[-1] * ones_like(x)

        def body(i, val):
            c0, c1, nd = val
            tmp = c0
            nd = nd - 1
            c0 = c[-i] - c1 * (2 * (nd - 1))
            c1 = tmp + c1 * x2
            return c0, c1, nd

        b = len(c) + 1
        x1 = (c0, c1, nd)
        y = x1
        for index in range(3, b):
            y = body(index, y)
        c0, c1, _ = y

    return c0 + c1 * x2


def hermval2d(x, y, c):
    return _evaluate(hermval, c, x, y)


def hermval3d(x, y, z, c):
    return _evaluate(hermval, c, x, y, z)


def hermvander(x, degree):
    if degree < 0:
        raise ValueError("degree must be non-negative")

    x = array(x, ndmin=1)
    dims = (degree + 1,) + x.shape
    dtyp = promote_types(x.dtype, array(0.0).dtype)
    x = x.astype(dtyp)
    v = empty(dims, dtype=dtyp)
    v = v.at[0].set(ones_like(x))
    if degree > 0:
        x2 = x * 2
        v = v.at[1].set(x2)

        def body(i, v):
            return v.at[i].set(v[i - 1] * x2 - v[i - 2] * (2 * (i - 1)))

        b = degree + 1
        y = v
        for index in range(2, b):
            y = body(index, y)
        v = y

    return moveaxis(v, 0, -1)


def hermvander2d(x, y, degree):
    return _vander_nd_flat(
        (hermvander, hermvander),
        (x, y),
        degree,
    )


def hermvander3d(x, y, z, degree):
    return _vander_nd_flat(
        (hermvander, hermvander, hermvander),
        (x, y, z),
        degree,
    )


def hermweight(x):
    return exp(-(x**2))


def lag2poly(c):
    c = _as_series(c)
    n = len(c)
    if n == 1:
        return c
    else:
        c0 = zeros_like(c).at[0].set(c[-2])
        c1 = zeros_like(c).at[0].set(c[-1])

        def body(k, c0c1):
            i = n - 1 - k
            c0, c1 = c0c1
            tmp = c0
            c0 = polysub(c[i - 2], (c1 * (i - 1)) / i)
            c1 = polyadd(tmp, polysub((2 * i - 1) * c1, polymulx(c1, "same")) / i)
            return c0, c1

        b = n - 2
        x = (c0, c1)
        y = x
        for index in range(0, b):
            y = body(index, y)
        c0, c1 = y

        return polyadd(c0, polysub(c1, polymulx(c1, "same")))


def lagadd(
    input: Num[Array, "..."],
    other: Num[Array, "..."],
) -> Num[Array, "..."]:
    return _add(input, other)


def lagcompanion(c):
    c = _as_series(c)

    if len(c) < 2:
        raise ValueError("Series must have maximum degree of at least 1.")

    if len(c) == 2:
        return array([[1 + c[0] / c[1]]])

    n = len(c) - 1
    mat = zeros((n, n), dtype=c.dtype).flatten()
    mat = mat.at[1 :: n + 1].set(-arange(1, n))
    mat = mat.at[0 :: n + 1].set(2.0 * arange(n) + 1.0)
    mat = mat.at[n :: n + 1].set(-arange(1, n))
    mat = mat.reshape((n, n))
    mat = mat.at[:, -1].add((c[:-1] / c[-1]) * n)
    return mat


def lagder(c, m=1, scl=1, axis=0):
    if m < 0:
        raise ValueError("The order of derivation must be non-negative")

    c = _as_series(c)

    if m == 0:
        return c

    c = moveaxis(c, axis, 0)
    n = len(c)
    if m >= n:
        c = zeros_like(c[:1])
    else:
        for _ in range(m):
            n = n - 1
            c *= scl
            der = empty((n,) + c.shape[1:], dtype=c.dtype)

            def body(k, der_c, n=n):
                j = n - k
                der, c = der_c
                der = der.at[j - 1].set(-c[j])
                c = c.at[j - 1].add(c[j])
                return der, c

            b = n - 1
            x = (der, c)
            y = x
            for index in range(0, b):
                y = body(index, y)
            der, c = y
            der = der.at[0].set(-c[1])
            c = der

    c = moveaxis(c, 0, axis)
    return c


def lagdiv(
    input: Num[Array, "..."],
    other: Num[Array, "..."],
) -> (Num[Array, "..."], Num[Array, "..."]):
    return _div(lagmul, input, other)


def lagfit(x, y, degree, rcond=None, full=False, w=None):
    return _fit(lagvander, x, y, degree, rcond, full, w)


def lagfromroots(roots):
    return _from_roots(lagline, lagmul, roots)


def laggauss(degree):
    degree = int(degree)
    if degree <= 0:
        raise ValueError("degree must be a positive integer")

    c = zeros(degree + 1).at[-1].set(1)
    m = lagcompanion(c)
    x = eigvalsh(m)

    dy = lagval(x, c)
    df = lagval(x, lagder(c))
    x -= dy / df

    fm = lagval(x, c[1:])
    fm /= abs(fm).max()
    df /= abs(df).max()
    w = 1 / (fm * df)

    w /= w.sum()

    return x, w


def laggrid2d(x, y, c):
    return _gridnd(lagval, c, x, y)


def laggrid3d(x, y, z, c):
    return _gridnd(lagval, c, x, y, z)


def lagint(c, m=1, k=None, lbnd=0, scl=1, axis=0):
    if k is None:
        k = []
    c = _as_series(c)
    lbnd, scl = map(asarray, (lbnd, scl))

    if not iterable(k):
        k = [k]
    if len(k) > m:
        raise ValueError("Too many integration constants")
    if ndim(lbnd) != 0:
        raise ValueError("lbnd must be a scalar.")
    if ndim(scl) != 0:
        raise ValueError("scl must be a scalar.")

    if m == 0:
        return c

    c = moveaxis(c, axis, 0)
    k = array(list(k) + [0] * (m - len(k)), ndmin=1)

    for i in range(m):
        n = len(c)
        c *= scl
        tmp = empty((n + 1,) + c.shape[1:], dtype=c.dtype)
        tmp = tmp.at[0].set(c[0])
        tmp = tmp.at[1].set(-c[0])
        j = arange(1, n)
        tmp = tmp.at[j].add(c[j])
        tmp = tmp.at[j + 1].add(-c[j])
        tmp = tmp.at[0].add(k[i] - lagval(lbnd, tmp))
        c = tmp

    c = moveaxis(c, 0, axis)
    return c


def lagline(off, scl):
    return array([off + scl, -scl])


def lagmul(input, other, mode="full"):
    input, other = _as_series(input, other)
    lc1, lc2 = len(input), len(other)

    if lc1 > lc2:
        c = other
        xs = input
    else:
        c = input
        xs = other

    if len(c) == 1:
        c0 = lagadd(zeros(lc1 + lc2 - 1), c[0] * xs)
        input = zeros(lc1 + lc2 - 1)
    elif len(c) == 2:
        c0 = lagadd(zeros(lc1 + lc2 - 1), c[0] * xs)
        input = lagadd(zeros(lc1 + lc2 - 1), c[1] * xs)
    else:
        nd = len(c)
        c0 = lagadd(zeros(lc1 + lc2 - 1), c[-2] * xs)
        input = lagadd(zeros(lc1 + lc2 - 1), c[-1] * xs)

        def body(i, val):
            c0, c1, nd = val
            tmp = c0
            nd = nd - 1
            c0 = lagsub(c[-i] * xs, (c1 * (nd - 1)) / nd)
            c1 = lagadd(tmp, lagsub((2 * nd - 1) * c1, lagmulx(c1, "same")) / nd)
            return c0, c1, nd

        b = len(c) + 1
        x = (c0, input, nd)
        y = x
        for index in range(3, b):
            y = body(index, y)
        c0, input, _ = y

    ret = lagadd(c0, lagsub(input, lagmulx(input, "same")))
    if mode == "same":
        ret = ret[: max(lc1, lc2)]
    return ret


def lagmulx(c, mode="full"):
    c = _as_series(c)

    prd = zeros(len(c) + 1, dtype=c.dtype)
    prd = prd.at[0].set(c[0])
    prd = prd.at[1].set(-c[0])

    i = arange(1, len(c))

    prd = prd.at[i + 1].set(-c[i] * (i + 1))
    prd = prd.at[i].add(c[i] * (2 * i + 1))
    prd = prd.at[i - 1].add(-c[i] * i)

    if mode == "same":
        prd = prd[: len(c)]
    return prd


def lagpow(c, pow, maxpower=16):
    return _pow(lagmul, c, pow, maxpower)


def lagroots(c):
    c = _as_series(c)

    if len(c) <= 1:
        return array([], dtype=c.dtype)

    if len(c) == 2:
        return array([1 + c[0] / c[1]])

    return sort(eigvals(lagcompanion(c)[::-1, ::-1]))


def lagsub(
    input: Num[Array, "..."],
    other: Num[Array, "..."],
) -> Num[Array, "..."]:
    return _subtract(input, other)


def lagval(x, c, tensor=True):
    c = _as_series(c)
    x = asarray(x)
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
        c0 = c[-2] * ones_like(x)
        c1 = c[-1] * ones_like(x)

        def body(i, val):
            c0, c1, nd = val
            tmp = c0
            nd = nd - 1
            c0 = c[-i] - (c1 * (nd - 1)) / nd
            c1 = tmp + (c1 * ((2 * nd - 1) - x)) / nd
            return c0, c1, nd

        b = len(c) + 1
        x1 = (c0, c1, nd)
        y = x1
        for index in range(3, b):
            y = body(index, y)
        c0, c1, _ = y

    return c0 + c1 * (1 - x)


def lagval2d(x, y, c):
    return _evaluate(lagval, c, x, y)


def lagval3d(x, y, z, c):
    return _evaluate(lagval, c, x, y, z)


def lagvander(x, degree):
    if degree < 0:
        raise ValueError("degree must be non-negative")

    x = array(x, ndmin=1)
    dims = (degree + 1,) + x.shape
    dtyp = promote_types(x.dtype, array(0.0).dtype)
    x = x.astype(dtyp)
    v = empty(dims, dtype=dtyp)
    v = v.at[0].set(ones_like(x))
    if degree > 0:
        v = v.at[1].set(1 - x)

        def body(i, v):
            return v.at[i].set((v[i - 1] * (2 * i - 1 - x) - v[i - 2] * (i - 1)) / i)

        b = degree + 1
        y = v
        for index in range(2, b):
            y = body(index, y)
        v = y

    return moveaxis(v, 0, -1)


def lagvander2d(x, y, degree):
    return _vander_nd_flat((lagvander, lagvander), (x, y), degree)


def lagvander3d(x, y, z, degree):
    return _vander_nd_flat((lagvander, lagvander, lagvander), (x, y, z), degree)


def lagweight(x):
    return exp(-x)


def leg2poly(c):
    c = _as_series(c)
    n = len(c)
    if n < 3:
        return c
    else:
        c0 = zeros_like(c).at[0].set(c[-2])
        c1 = zeros_like(c).at[0].set(c[-1])

        def body(k, c0c1):
            i = n - 1 - k
            c0, c1 = c0c1
            tmp = c0
            c0 = polysub(c[i - 2], c1 * (i - 1) / i)
            c1 = polyadd(tmp, polymulx(c1, "same") * (2 * i - 1) / i)
            return c0, c1

        b = n - 2
        x = (c0, c1)
        y = x
        for index in range(0, b):
            y = body(index, y)
        c0, c1 = y

        return polyadd(c0, polymulx(c1, "same"))


def legadd(
    input: Num[Array, "..."],
    other: Num[Array, "..."],
) -> Num[Array, "..."]:
    return _add(input, other)


def legcompanion(c):
    c = _as_series(c)
    if len(c) < 2:
        raise ValueError("Series must have maximum degree of at least 1.")
    if len(c) == 2:
        return array([[-c[0] / c[1]]])

    n = len(c) - 1
    mat = zeros((n, n), dtype=c.dtype)
    scl = 1.0 / sqrt(2 * arange(n) + 1)
    shp = mat.shape
    mat = mat.flatten()
    mat = mat.at[1 :: n + 1].set(arange(1, n) * scl[: n - 1] * scl[1:n])
    mat = mat.at[n :: n + 1].set(arange(1, n) * scl[: n - 1] * scl[1:n])
    mat = mat.reshape(shp)
    mat = mat.at[:, -1].add(-(c[:-1] / c[-1]) * (scl / scl[-1]) * (n / (2 * n - 1)))
    return mat


def legder(c, m=1, scl=1, axis=0):
    if m < 0:
        raise ValueError("The order of derivation must be non-negative")

    c = _as_series(c)

    if m == 0:
        return c

    c = moveaxis(c, axis, 0)
    n = len(c)
    if m >= n:
        c = zeros_like(c[:1])
    else:
        for _ in range(m):
            n = n - 1
            c *= scl
            der = empty((n,) + c.shape[1:], dtype=c.dtype)

            def body(k, der_c, n=n):
                j = n - k
                der, c = der_c
                der = der.at[j - 1].set((2 * j - 1) * c[j])
                c = c.at[j - 2].add(c[j])
                return der, c

            b = n - 2
            x = (der, c)
            y = x
            for index in range(0, b):
                y = body(index, y)
            der, c = y
            if n > 1:
                der = der.at[1].set(3 * c[2])
            der = der.at[0].set(c[1])
            c = der

    c = moveaxis(c, 0, axis)
    return c


def legdiv(
    input: Num[Array, "..."],
    other: Num[Array, "..."],
) -> (Num[Array, "..."], Num[Array, "..."]):
    return _div(legmul, input, other)


def legfit(x, y, degree, rcond=None, full=False, w=None):
    return _fit(legvander, x, y, degree, rcond, full, w)


def legfromroots(roots):
    return _from_roots(legline, legmul, roots)


def leggauss(degree):
    degree = int(degree)
    if degree <= 0:
        raise ValueError("degree must be a positive integer")

    c = zeros(degree + 1).at[-1].set(1)
    m = legcompanion(c)
    x = eigvalsh(m)

    dy = legval(x, c)
    df = legval(x, legder(c))
    x -= dy / df

    fm = legval(x, c[1:])
    fm /= abs(fm).max()
    df /= abs(df).max()
    w = 1 / (fm * df)

    w = (w + w[::-1]) / 2
    x = (x - x[::-1]) / 2

    w *= 2.0 / w.sum()

    return x, w


def leggrid2d(x, y, c):
    return _gridnd(legval, c, x, y)


def leggrid3d(x, y, z, c):
    return _gridnd(legval, c, x, y, z)


def legint(c, m=1, k=None, lbnd=0, scl=1, axis=0):
    if k is None:
        k = []
    c = _as_series(c)
    lbnd, scl = map(asarray, (lbnd, scl))

    if not iterable(k):
        k = [k]
    if len(k) > m:
        raise ValueError("Too many integration constants")
    if ndim(lbnd) != 0:
        raise ValueError("lbnd must be a scalar.")
    if ndim(scl) != 0:
        raise ValueError("scl must be a scalar.")

    if m == 0:
        return c

    c = moveaxis(c, axis, 0)
    k = array(list(k) + [0] * (m - len(k)), ndmin=1)

    for i in range(m):
        n = len(c)
        c *= scl
        tmp = empty((n + 1,) + c.shape[1:], dtype=c.dtype)
        tmp = tmp.at[0].set(c[0] * 0)
        tmp = tmp.at[1].set(c[0])
        if n > 1:
            tmp = tmp.at[2].set(c[1] / 3)
        j = arange(2, n)
        t = (c[j].T / (2 * j + 1)).T
        tmp = tmp.at[j + 1].set(t)
        tmp = tmp.at[j - 1].add(-t)
        tmp = tmp.at[0].add(k[i] - legval(lbnd, tmp))
        c = tmp
    c = moveaxis(c, 0, axis)
    return c


def legline(off, scl):
    return array([off, scl])


def legmul(input, other, mode="full"):
    input, other = _as_series(input, other)
    lc1, lc2 = len(input), len(other)
    if lc1 > lc2:
        c = other
        xs = input
    else:
        c = input
        xs = other

    if len(c) == 1:
        c0 = legadd(zeros(lc1 + lc2 - 1), c[0] * xs)
        input = zeros(lc1 + lc2 - 1)
    elif len(c) == 2:
        c0 = legadd(zeros(lc1 + lc2 - 1), c[0] * xs)
        input = legadd(zeros(lc1 + lc2 - 1), c[1] * xs)
    else:
        nd = len(c)
        c0 = legadd(zeros(lc1 + lc2 - 1), c[-2] * xs)
        input = legadd(zeros(lc1 + lc2 - 1), c[-1] * xs)

        def body(i, val):
            c0, c1, nd = val
            tmp = c0
            nd = nd - 1
            c0 = legsub(c[-i] * xs, (c1 * (nd - 1)) / nd)
            c1 = legadd(tmp, (legmulx(c1, "same") * (2 * nd - 1)) / nd)
            return c0, c1, nd

        b = len(c) + 1
        x = (c0, input, nd)
        y = x
        for index in range(3, b):
            y = body(index, y)
        c0, input, _ = y

    ret = legadd(c0, legmulx(input, "same"))
    if mode == "same":
        ret = ret[: max(lc1, lc2)]
    return ret


def legmulx(c, mode="full"):
    c = _as_series(c)
    prd = zeros(len(c) + 1, dtype=c.dtype)
    prd = prd.at[1].set(c[0])

    def body(i, prd):
        j = i + 1

        k = i - 1

        s = i + j

        prd = prd.at[j].set((c[i] * j) / s)

        prd = prd.at[k].add((c[i] * i) / s)

        return prd

    b = len(c)
    y = prd
    for index in range(1, b):
        y = body(index, y)
    prd = y

    if mode == "same":
        prd = prd[: len(c)]

    return prd


def legpow(c, pow, maxpower=16):
    return _pow(legmul, c, pow, maxpower)


def legroots(c):
    c = _as_series(c)

    if len(c) <= 1:
        return array([], dtype=c.dtype)

    if len(c) == 2:
        return array([-c[0] / c[1]])

    return sort(eigvals(legcompanion(c)[::-1, ::-1]))


def legsub(
    input: Num[Array, "..."],
    other: Num[Array, "..."],
) -> Num[Array, "..."]:
    return _subtract(input, other)


def legval(x, c, tensor=True):
    c = _as_series(c)
    x = asarray(x)
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
        c0 = c[-2] * ones_like(x)
        c1 = c[-1] * ones_like(x)

        def body(i, val):
            c0, c1, nd = val
            tmp = c0
            nd = nd - 1
            c0 = c[-i] - (c1 * (nd - 1)) / nd
            c1 = tmp + (c1 * x * (2 * nd - 1)) / nd
            return c0, c1, nd

        b = len(c) + 1
        x1 = (c0, c1, nd)
        y = x1
        for index in range(3, b):
            y = body(index, y)
        c0, c1, _ = y

    return c0 + c1 * x


def legval2d(x, y, c):
    return _evaluate(legval, c, x, y)


def legval3d(x, y, z, c):
    return _evaluate(legval, c, x, y, z)


def legvander(x, degree):
    if degree < 0:
        raise ValueError("degree must be non-negative")

    x = array(x, ndmin=1)

    dims = (degree + 1,) + x.shape

    dtyp = promote_types(x.dtype, array(0.0).dtype)

    x = x.astype(dtyp)

    v = empty(dims, dtype=dtyp)

    v = v.at[0].set(ones_like(x))

    if degree > 0:
        v = v.at[1].set(x)

        def body(i, v):
            return v.at[i].set((v[i - 1] * x * (2 * i - 1) - v[i - 2] * (i - 1)) / i)

        b = degree + 1
        y = v
        for index in range(2, b):
            y = body(index, y)
        v = y

    return moveaxis(v, 0, -1)


def legvander2d(x, y, degree):
    return _vander_nd_flat((legvander, legvander), (x, y), degree)


def legvander3d(x, y, z, degree):
    return _vander_nd_flat((legvander, legvander, legvander), (x, y, z), degree)


def legweight(x):
    return ones_like(x)


def _map_domain(x, old, new):
    x = asarray(x)

    off, scl = _map_parameters(old, new)

    return off + scl * x


def _map_parameters(old, new):
    oldlen = old[1] - old[0]

    newlen = new[1] - new[0]

    off = (old[1] * new[0] - old[0] * new[1]) / oldlen

    scl = newlen / oldlen

    return off, scl


def poly2cheb(pol):
    pol = _as_series(pol)

    degree = len(pol) - 1

    res = zeros_like(pol)

    def body(i, res):
        return chebadd(chebmulx(res, mode="same"), pol[(degree - i)])

    b = degree + 1
    y = res
    for index in range(0, b):
        y = body(index, y)
    return y


def poly2herm(pol):
    pol = _as_series(pol)

    degree = len(pol) - 1

    res = zeros_like(pol)

    def body(i, res):
        return hermadd(hermmulx(res, mode="same"), pol[(degree - i)])

    b = degree + 1
    y = res
    for index in range(0, b):
        y = body(index, y)
    return y


def poly2herme(pol):
    pol = _as_series(pol)

    degree = len(pol) - 1

    res = zeros_like(pol)

    def body(i, res):
        return hermeadd(hermemulx(res, mode="same"), pol[(degree - i)])

    b = degree + 1
    y = res
    for index in range(0, b):
        y = body(index, y)
    return y


def poly2lag(pol):
    pol = _as_series(pol)

    res = zeros_like(pol)

    def body(i, res):
        res = lagadd(lagmulx(res, mode="same"), pol[::-1][i])

        return res

    b = len(pol)
    y = res
    for index in range(0, b):
        y = body(index, y)
    return y


def poly2leg(pol):
    pol = _as_series(pol)

    degree = len(pol) - 1

    res = zeros_like(pol)

    def body(i, res):
        k = degree - i

        res = legadd(legmulx(res, mode="same"), pol[k])

        return res

    b = degree + 1
    y = res
    for index in range(0, b):
        y = body(index, y)
    return y


def polyadd(
    input: Num[Array, "..."],
    other: Num[Array, "..."],
) -> Num[Array, "..."]:
    return _add(input, other)


def polycompanion(c):
    c = _as_series(c)

    if len(c) < 2:
        raise ValueError("Series must have maximum degree of at least 1.")

    if len(c) == 2:
        return array([[-c[0] / c[1]]])

    n = len(c) - 1

    mat = zeros((n, n), dtype=c.dtype).flatten()

    mat = mat.at[n :: n + 1].set(1)

    mat = mat.reshape((n, n))

    return mat.at[:, -1].add(-c[:-1] / c[-1])


def polyder(c, m=1, scl=1, axis=0):
    c = _as_series(c)

    if m == 0:
        return c

    c = moveaxis(c, axis, 0)

    n = c.shape[0]

    if m >= n:
        c = zeros_like(c[:1])
    else:
        D = arange(n)

        def body(i, c):
            c = (D * c.T).T

            c = roll(c, -1, axis=0) * scl

            c = c.at[-1].set(0)

            return c

        y = c
        for index in range(0, m):
            y = body(index, y)
        c = y

        c = c[:-m]

    c = moveaxis(c, 0, axis)

    return c


def polydiv(
    input: Num[Array, "..."],
    other: Num[Array, "..."],
) -> (Num[Array, "..."], Num[Array, "..."]):
    input, other = _as_series(input, other)

    return _div(polymul, input, other)


def polyfit(x, y, degree, rcond=None, full=False, w=None):
    return _fit(polyvander, x, y, degree, rcond, full, w)


def polyfromroots(roots):
    return _from_roots(polyline, polymul, roots)


def polygrid2d(x, y, c):
    return _gridnd(polyval, c, x, y)


def polygrid3d(x, y, z, c):
    return _gridnd(polyval, c, x, y, z)


def polyint(c, m=1, k=None, lbnd=0, scl=1, axis=0):
    if k is None:
        k = []

    c = _as_series(c)

    lbnd, scl = map(asarray, (lbnd, scl))

    if not iterable(k):
        k = [k]

    if m < 0:
        raise ValueError("The order of integration must be non-negative")

    if len(k) > m:
        raise ValueError("Too many integration constants")

    if ndim(lbnd) != 0:
        raise ValueError("lbnd must be a scalar.")

    if ndim(scl) != 0:
        raise ValueError("scl must be a scalar.")

    if m == 0:
        return c

    k = array(list(k) + [0] * (m - len(k)), ndmin=1)

    n = c.shape[axis]

    c = _pad_along_axis(c, (0, m), axis)

    c = moveaxis(c, axis, 0)

    D = arange(n + m) + 1

    def body(i, c):
        c *= scl

        c = (c.T / D).T  # broadcasting correctly

        c = roll(c, 1, axis=0)

        c = c.at[0].set(0)

        offset = k[i] - polyval(lbnd, c)

        c = c.at[0].add(offset)

        return c

    y = c
    for index in range(0, m):
        y = body(index, y)
    c = y

    c = moveaxis(c, 0, axis)

    return c


def polyline(off, scl):
    return array([off, scl])


def polymul(input, other, mode="full"):
    input, other = _as_series(input, other)

    ret = convolve(input, other)

    if mode == "same":
        ret = ret[: max(len(input), len(other))]

    return ret


def polymulx(c, mode="full"):
    c = _as_series(c)

    prd = zeros(len(c) + 1, dtype=c.dtype)

    prd = prd.at[1:].set(c)

    if mode == "same":
        prd = prd[: len(c)]

    return prd


def polypow(c, pow, maxpower=16):
    return _pow(polymul, c, pow, maxpower)


def polyroots(c):
    c = _as_series(c)

    if len(c) < 2:
        return array([], dtype=c.dtype)

    if len(c) == 2:
        return array([-c[0] / c[1]])

    return sort(eigvals(polycompanion(c)[::-1, ::-1]))


def polysub(
    input: Num[Array, "..."],
    other: Num[Array, "..."],
) -> Num[Array, "..."]:
    return _subtract(input, other)


def polyval(x, c, tensor=True):
    c = _as_series(c)

    x = asarray(x)

    if tensor:
        c = c.reshape(c.shape + (1,) * x.ndim)

    c0 = c[-1] + zeros_like(x)

    def body(i, c0):
        c0 = c[-i] + c0 * x

        return c0

    b = len(c) + 1
    y = c0
    for index in range(2, b):
        y = body(index, y)
    return y


def polyval2d(x, y, c):
    return _evaluate(polyval, c, x, y)


def polyval3d(x, y, z, c):
    return _evaluate(polyval, c, x, y, z)


def polyvalfromroots(x, r, tensor=True):
    r = array(r, ndmin=1)

    x = asarray(x)

    if tensor:
        r = r.reshape(r.shape + (1,) * x.ndim)

    if x.ndim >= r.ndim:
        raise ValueError("x.ndim must be < r.ndim when tensor == False")

    return prod(x - r, axis=0)


def polyvander(x, degree):
    if degree < 0:
        raise ValueError("degree must be non-negative")

    x = array(x, ndmin=1)

    dims = (degree + 1,) + x.shape

    dtyp = x.dtype

    v = empty(dims, dtype=dtyp)

    v = v.at[0].set(ones_like(x))

    def func(i, v):
        return v.at[i].set(v[i - 1] * x)

    b = degree + 1
    y = v
    for index in range(1, b):
        y = func(index, y)
    v = y

    output = moveaxis(v, 0, -1)

    return output


def polyvander2d(x, y, degree):
    return _vander_nd_flat(
        (polyvander, polyvander),
        (x, y),
        degree,
    )


def polyvander3d(x, y, z, degree):
    return _vander_nd_flat(
        (polyvander, polyvander, polyvander),
        (x, y, z),
        degree,
    )


def _trim_coefficients(c, tol=0):
    if tol < 0:
        raise ValueError("tol must be non-negative")

    c = _as_series(c)

    [ind] = nonzero(abs(c) > tol)

    if len(ind) == 0:
        return c[:1] * 0
    else:
        return c[: ind[-1] + 1].copy()


def _trim_sequence(seq):
    if len(seq) == 0:
        return seq
    else:
        for i in range(len(seq) - 1, -1, -1):
            if seq[i] != 0:
                break

        return seq[: i + 1]


chebtrim = _trim_coefficients
hermetrim = _trim_coefficients
hermtrim = _trim_coefficients
lagtrim = _trim_coefficients
legtrim = _trim_coefficients
polytrim = _trim_coefficients

__all__ = [
    "_map_domain",
    "_map_parameters",
    "_trim_coefficients",
    "_trim_sequence",
    "_as_series",
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
]
