import functools
import math
import operator
from typing import Literal

import jax
import jax.numpy
from jax import Array
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
    nonzero,
    ones,
    ones_like,
    pad,
    prod,
    promote_types,
    reshape,
    roll,
    sin,
    sort,
    sqrt,
    square,
    stack,
    sum,
    transpose,
    where,
    zeros,
    zeros_like,
)
from jax.numpy.linalg import (
    eigvals,
    eigvalsh,
    lstsq,
)

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


def _add(input, other):
    input, other = _as_series(input, other)

    if len(input) > len(other):
        output = input.at[: math.prod(other.shape)].add(other)
    else:
        output = other.at[: math.prod(input.shape)].add(input)

    return output


def _c_series_to_z_series(input):
    n = math.prod(input.shape)

    zs = zeros(2 * n - 1, dtype=input.dtype)

    zs = zs.at[n - 1 :].set(input / 2)

    output = zs + zs[::-1]

    return output


def _div(func, input, other):
    input, other = _as_series(input, other)

    lc1 = len(input)
    lc2 = len(other)

    if lc1 < lc2:
        return zeros_like(input[:1]), input

    if lc2 == 1:
        return input / other[-1], zeros_like(input[:1])

    def _ldordidx(x):
        return len(x) - 1 - nonzero(x[::-1], size=1)[0][0]

    quotient = zeros(lc1 - lc2 + 1, dtype=input.dtype)

    ridx = len(input) - 1

    sz = lc1 - _ldordidx(other) - 1

    y = zeros(lc1 + lc2 + 1, dtype=input.dtype).at[sz].set(1.0)

    y1 = quotient, input, y, ridx

    for index in range(0, sz):
        quotient, remainder, y2, ridx1 = y1

        i = sz - index

        p = func(y2, other)

        pidx = _ldordidx(p)

        t = remainder[ridx1] / p[pidx]

        remainder = _subtract(
            remainder.at[ridx1].set(0),
            t * p.at[pidx].set(0),
        )[: len(remainder)]

        quotient = quotient.at[i].set(t)

        ridx1 = ridx1 - 1

        y2 = roll(y2, -1)

        y1 = quotient, remainder, y2, ridx1

    quotient, remainder, _, _ = y1

    return quotient, remainder


def _fit(
    vandermonde_func,
    input: Array,
    other: Array,
    degree: Array | int,
    relative_condition: float | None = None,
    full: bool = False,
    weight: Array | None = None,
):
    degree = asarray(degree)

    if degree.ndim > 1 or degree.dtype.kind not in "iu" or math.prod(degree.shape) == 0:
        raise TypeError

    if degree.min() < 0:
        raise ValueError

    if input.ndim != 1:
        raise TypeError

    if math.prod(input.shape) == 0:
        raise TypeError

    if other.ndim < 1 or other.ndim > 2:
        raise TypeError

    if input.shape[0] != other.shape[0]:
        raise TypeError

    if degree.ndim == 0:
        maximum = int(degree)

        vandermonde = vandermonde_func(input, maximum)
    else:
        degree = sort(degree)

        maximum = int(degree[-1])

        vandermonde = vandermonde_func(input, maximum)[:, degree]

    lhs = transpose(vandermonde)
    rhs = transpose(other)

    if weight is not None:
        weight = asarray(weight)

        if weight.ndim != 1:
            raise TypeError

        if input.shape[0] != weight.shape[0]:
            raise TypeError

        lhs = lhs * weight
        rhs = rhs * weight

    if relative_condition is None:
        relative_condition = input.shape[0] * finfo(input.dtype).eps

    if issubclass(lhs.dtype.type, complexfloating):
        scale = square(lhs.real) + square(lhs.imag)

        scale = sum(scale, 1)

        scale = sqrt(scale)
    else:
        scale = square(lhs)

        scale = sum(scale, 1)

        scale = sqrt(scale)

    scale = where(scale == 0, 1, scale)

    output, residuals, rank, s = lstsq(
        transpose(lhs) / scale,
        transpose(rhs),
        relative_condition,
    )

    output = transpose(transpose(output) / scale)

    if degree.ndim > 0:
        if output.ndim == 2:
            output = (
                zeros((maximum + 1, output.shape[1]), dtype=output.dtype)
                .at[degree]
                .set(output)
            )
        else:
            output = zeros(maximum + 1, dtype=output.dtype).at[degree].set(output)

    if full:
        return output, [residuals, rank, s, relative_condition]
    else:
        return output


def _from_roots(f, g, input):
    if math.prod(input.shape) == 0:
        return ones([1])

    input = sort(input)

    ys = []

    for x in input:
        y = _add(zeros(input.shape[0] + 1, dtype=x.dtype), f(-x, 1))

        ys = [*ys, y]

    p = stack(ys)

    n = p.shape[0]

    val = n, p

    while val[0] > 1:
        m, r = divmod(val[0], 2)

        arr = val[1]

        tmp = array([zeros(input.shape[0] + 1, dtype=p.dtype)] * len(p))

        val1 = tmp

        for i in range(0, m):
            val1 = val1.at[i].set(g(arr[i], arr[i + m])[: input.shape[0] + 1])

        tmp = val1

        if r:
            tmp = tmp.at[0].set(g(tmp[0], arr[2 * m])[: input.shape[0] + 1])

        val = m, tmp

    _, ret = val

    return ret[0]


def _normed_hermite_e_n(x, n):
    if n == 0:
        output = full(x.shape, 1 / sqrt(sqrt(2 * math.pi)))
    else:
        a = zeros_like(x)
        b = ones_like(x) / sqrt(sqrt(2 * math.pi))

        d = array(n).astype(float)

        for _ in range(0, n - 1):
            c = a

            a = -b * sqrt((d - 1.0) / d)

            b = c + b * x * sqrt(1.0 / d)

            d = d - 1.0

        output = a + b * x

    return output


def _normed_hermite_n(x, n):
    if n == 0:
        output = full(x.shape, 1 / sqrt(sqrt(math.pi)))
    else:
        c0 = zeros_like(x)
        c1 = ones_like(x) / sqrt(sqrt(math.pi))
        nd = array(n).astype(float)

        for _ in range(0, n - 1):
            tmp = c0

            c0 = -c1 * sqrt((nd - 1.0) / nd)

            c1 = tmp + c1 * x * sqrt(2.0 / nd)

            nd = nd - 1.0

        output = c0 + c1 * x * sqrt(2)

    return output


def _nth_slice(i, ndim):
    sl = [None] * ndim
    sl[i] = slice(None)
    return tuple(sl)


def _pad_along_axis(input, padding=(0, 0), axis=0):
    input = moveaxis(input, axis, 0)

    if padding[0] < 0:
        input = input[abs(padding[0]) :]
        padding = (0, padding[1])
    if padding[1] < 0:
        input = input[: -abs(padding[1])]
        padding = (padding[0], 0)

    npad = [(0, 0)] * input.ndim
    npad[0] = padding

    output = pad(input, pad_width=npad, mode="constant", constant_values=0)
    return moveaxis(output, 0, axis)


def _pow(func, input, exponent, maximum_exponent):
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


def _subtract(input, other):
    input, other = _as_series(input, other)

    if len(input) > len(other):
        output = -other

        output = input.at[: math.prod(other.shape)].add(output)

        return output

    output = -other

    output = output.at[: math.prod(input.shape)].add(input)

    return output


def _evaluate(func, input, *args):
    if not all(a.shape == args[0].shape for a in args[1:]):
        match len(args):
            case 2:
                raise ValueError
            case 3:
                raise ValueError
            case _:
                raise ValueError

    args = iter(args)

    output = func(next(args), input)

    for x in args:
        output = func(x, output, tensor=False)

    return output


def _vander_nd(vander_fs, points, degrees):
    n_dims = len(vander_fs)
    if n_dims != len(points):
        raise ValueError
    if n_dims != len(degrees):
        raise ValueError
    if n_dims == 0:
        raise ValueError

    points = tuple(array(tuple(points), copy=False) + 0.0)

    vander_arrays = (
        vander_fs[i](points[i], degrees[i])[(...,) + _nth_slice(i, n_dims)]
        for i in range(n_dims)
    )

    return functools.reduce(operator.mul, vander_arrays)


def _vander_nd_flat(vander_fs, points, degrees):
    v = _vander_nd(vander_fs, points, degrees)
    return reshape(v, v.shape[: -len(degrees)] + (-1,))


def _z_series_mul(z1, z2, mode="full"):
    return convolve(z1, z2, mode=mode)


def _z_series_to_c_series(zs):
    n = (math.prod(zs.shape) + 1) // 2
    c = zs[n - 1 :].copy()
    return c.at[1:n].multiply(2)


def _as_series(*arrs, trim: bool = False):
    arrays = [array(a, ndmin=1) for a in arrs]

    if trim:
        arrays = [_trim_sequence(a) for a in arrays]

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

    for i in range(0, n - 2):
        i1 = n - 1 - i

        tmp = c0

        c0 = polysub(c[i1 - 2], c1)

        c1 = polyadd(tmp, polymulx(c1, "same") * 2)

    output = polymulx(c1, "same")

    output = polyadd(c0, output)

    return output


def chebadd(input, other):
    return _add(input, other)


def chebcompanion(c):
    c = _as_series(c)
    if len(c) < 2:
        raise ValueError
    if len(c) == 2:
        return array([[-c[0] / c[1]]])

    n = len(c) - 1

    mat = zeros((n, n), dtype=c.dtype)

    scl = ones(n).at[1:].set(sqrt(0.5))

    shp = mat.shape

    mat = reshape(mat, [-1])

    mat = mat.at[1 :: n + 1].set(full(n - 1, 1 / 2).at[0].set(sqrt(0.5)))
    mat = mat.at[n :: n + 1].set(full(n - 1, 1 / 2).at[0].set(sqrt(0.5)))

    mat = reshape(mat, shp)

    return mat.at[:, -1].add(-(c[:-1] / c[-1]) * (scl / scl[-1]) * 0.5)


def chebder(input, order=1, scale=1, axis=0):
    if order < 0:
        raise ValueError

    input = _as_series(input)

    if order == 0:
        return input

    output = moveaxis(input, axis, 0)

    n = len(output)

    if order >= n:
        output = zeros_like(output[:1])
    else:
        for _ in range(order):
            n = n - 1

            output = output * scale

            derivative = empty((n,) + output.shape[1:], dtype=output.dtype)

            for i in range(0, n - 2):
                j = n - i

                derivative = derivative.at[j - 1].set((2 * j) * output[j])

                output = output.at[j - 2].add((j * output[j]) / (j - 2))

            if n > 1:
                derivative = derivative.at[1].set(4 * output[2])

            output = derivative.at[0].set(output[1])

    return moveaxis(output, 0, axis)


def chebdiv(input, other):
    return _div(chebmul, input, other)


def chebfit(
    input: Array,
    other: Array,
    degree: Array | int,
    relative_condition: float | None = None,
    full: bool = False,
    weight: Array | None = None,
):
    return _fit(
        chebvander,
        input,
        other,
        degree,
        relative_condition,
        full,
        weight,
    )


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
    for arg in [x, y]:
        c = chebval(arg, c)
    return c


def chebgrid3d(x, y, z, c):
    for arg in [x, y, z]:
        c = chebval(arg, c)
    return c


def chebint(c, order=1, k=None, lbnd=0, scl=1, axis=0):
    if k is None:
        k = []
    c = _as_series(c)
    lbnd, scl = map(asarray, (lbnd, scl))

    if not iterable(k):
        k = [k]
    if len(k) > order:
        raise ValueError
    if ndim(lbnd) != 0:
        raise ValueError
    if ndim(scl) != 0:
        raise ValueError

    if order == 0:
        return c

    c = moveaxis(c, axis, 0)
    k = array(list(k) + [0] * (order - len(k)), ndmin=1)

    for i in range(order):
        n = len(c)
        c *= scl
        tmp = empty((n + 1,) + c.shape[1:], dtype=c.dtype)
        tmp = tmp.at[0].set(c[0] * 0)
        tmp = tmp.at[1].set(c[0])
        if n > 1:
            tmp = tmp.at[2].set(c[1] / 4)
        j = arange(2, n)
        tmp = tmp.at[j + 1].set(transpose(transpose(c[j]) / (2 * (j + 1))))
        tmp = tmp.at[j - 1].add(-transpose(transpose(c[j]) / (2 * (j - 1))))
        tmp = tmp.at[0].add(k[i] - chebval(lbnd, tmp))
        c = tmp
    c = moveaxis(c, 0, axis)
    return c


def chebinterpolate(func, degree, args=()):
    _deg = int(degree)
    if _deg != degree:
        raise ValueError
    if _deg < 0:
        raise ValueError

    order = _deg + 1
    xcheb = chebpts1(order)
    yfunc = func(xcheb, *args)
    m = chebvander(xcheb, _deg)
    c = dot(transpose(m), yfunc)
    c = c.at[0].divide(order)
    c = c.at[1:].divide(0.5 * order)

    return c


def chebline(off, scl):
    return array([off, scl])


def chebmul(
    input,
    other,
    mode: Literal["full", "same"] = "full",
):
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

    y = output

    for _ in range(2, power + 1):
        y = convolve(y, zs, mode="same")

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


def chebsub(input, other):
    return _subtract(input, other)


def chebval(x, c, tensor=True):
    c = _as_series(c)

    if tensor:
        c = reshape(c, c.shape + (1,) * x.ndim)

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

        x1 = (c0, c1)

        for index in range(3, c.shape[0] + 1):
            x1 = body(index, x1)

        c0, c1 = x1

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

        for i in range(2, degree + 1):
            v = v.at[i].set(v[i - 1] * x2 - v[i - 2])

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
    output = sqrt(1.0 + x) * sqrt(1.0 - x)

    output = 1.0 / output

    return output


def _get_domain(x: Array) -> Array:
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
    input,
    other,
):
    return _add(input, other)


def hermcompanion(c):
    c = _as_series(c)
    if len(c) < 2:
        raise ValueError
    if len(c) == 2:
        return array([[-0.5 * c[0] / c[1]]])

    n = len(c) - 1
    mat = zeros((n, n), dtype=c.dtype)
    scl = hstack((1.0, 1.0 / sqrt(2.0 * arange(n - 1, 0, -1))))
    scl = cumprod(scl)[::-1]
    shp = mat.shape
    mat = reshape(mat, [-1])
    mat = mat.at[1 :: n + 1].set(sqrt(0.5 * arange(1, n)))
    mat = mat.at[n :: n + 1].set(sqrt(0.5 * arange(1, n)))
    mat = reshape(mat, shp)
    mat = mat.at[:, -1].add(-scl * c[:-1] / (2.0 * c[-1]))
    return mat


def hermder(c, order=1, scl=1, axis=0):
    if order < 0:
        raise ValueError

    c = _as_series(c)

    if order == 0:
        return c

    c = moveaxis(c, axis, 0)
    n = len(c)
    if order >= n:
        c = zeros_like(c[:1])
    else:
        for _ in range(order):
            n = n - 1
            c *= scl
            der = empty((n,) + c.shape[1:], dtype=c.dtype)
            j = arange(n, 0, -1)
            der = der.at[j - 1].set(transpose(2 * j * transpose(c[j])))
            c = der
    c = moveaxis(c, 0, axis)
    return c


def hermdiv(
    input,
    other,
):
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
    input,
    other,
):
    return _add(input, other)


def hermecompanion(c):
    c = _as_series(c)
    if len(c) < 2:
        raise ValueError
    if len(c) == 2:
        return array([[-c[0] / c[1]]])

    n = len(c) - 1
    mat = zeros((n, n), dtype=c.dtype)
    scl = hstack((1.0, 1.0 / sqrt(arange(n - 1, 0, -1))))
    scl = cumprod(scl)[::-1]
    shp = mat.shape
    mat = reshape(mat, [-1])
    mat = mat.at[1 :: n + 1].set(sqrt(arange(1, n)))
    mat = mat.at[n :: n + 1].set(sqrt(arange(1, n)))
    mat = reshape(mat, shp)
    mat = mat.at[:, -1].add(-scl * c[:-1] / c[-1])
    return mat


def hermeder(c, order=1, scl=1, axis=0):
    if order < 0:
        raise ValueError

    c = _as_series(c)

    if order == 0:
        return c

    c = moveaxis(c, axis, 0)
    n = len(c)
    if order >= n:
        c = zeros_like(c[:1])
    else:
        for _ in range(order):
            n = n - 1
            c *= scl
            der = empty((n,) + c.shape[1:], dtype=c.dtype)
            j = arange(n, 0, -1)
            der = der.at[j - 1].set(transpose(j * transpose(c[j])))
            c = der
    c = moveaxis(c, 0, axis)
    return c


def hermediv(
    input,
    other,
):
    return _div(hermemul, input, other)


def hermefit(
    input: Array,
    other: Array,
    degree: Array | int,
    relative_condition: float | None = None,
    full: bool = False,
    weight: Array | None = None,
):
    return _fit(
        hermevander,
        input,
        other,
        degree,
        relative_condition,
        full,
        weight,
    )


def hermefromroots(roots):
    return _from_roots(hermeline, hermemul, roots)


def hermegauss(degree):
    degree = int(degree)
    if degree <= 0:
        raise ValueError

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
    for arg in [x, y]:
        c = hermeval(arg, c)
    return c


def hermegrid3d(x, y, z, c):
    for arg in [x, y, z]:
        c = hermeval(arg, c)
    return c


def hermeint(c, order=1, k=None, lbnd=0, scl=1, axis=0):
    if k is None:
        k = []

    c = _as_series(c)

    lbnd, scl = map(asarray, (lbnd, scl))

    if not iterable(k):
        k = [k]

    if len(k) > order:
        raise ValueError

    if ndim(lbnd) != 0:
        raise ValueError

    if ndim(scl) != 0:
        raise ValueError

    if order == 0:
        return c

    c = moveaxis(c, axis, 0)
    k = array(list(k) + [0] * (order - len(k)), ndmin=1)

    for i in range(order):
        n = len(c)
        c *= scl
        tmp = empty((n + 1,) + c.shape[1:], dtype=c.dtype)
        tmp = tmp.at[0].set(c[0] * 0)
        tmp = tmp.at[1].set(c[0])
        j = arange(1, n)
        tmp = tmp.at[j + 1].set(transpose(transpose(c[j]) / (j + 1)))
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
    input,
    other,
):
    return _subtract(input, other)


def hermeval(x, c, tensor=True):
    c = _as_series(c)

    if tensor:
        c = reshape(c, c.shape + (1,) * x.ndim)

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
        raise ValueError

    x = array(x, ndmin=1)
    dims = (degree + 1,) + x.shape
    dtyp = promote_types(x.dtype, array(0.0).dtype)
    x = x.astype(dtyp)
    v = empty(dims, dtype=dtyp)
    v = v.at[0].set(ones_like(x))
    if degree > 0:
        v = v.at[1].set(x)

        for i in range(2, degree + 1):
            v = v.at[i].set(v[i - 1] * x - v[i - 2] * (i - 1))

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


def hermfit(
    input: Array,
    other: Array,
    degree: Array | int,
    relative_condition: float | None = None,
    full: bool = False,
    weight: Array | None = None,
):
    return _fit(
        hermvander,
        input,
        other,
        degree,
        relative_condition,
        full,
        weight,
    )


def hermfromroots(roots):
    return _from_roots(hermline, hermmul, roots)


def hermgauss(degree):
    degree = int(degree)
    if degree <= 0:
        raise ValueError

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
    for arg in [x, y]:
        c = hermval(arg, c)
    return c


def hermgrid3d(x, y, z, c):
    for arg in [x, y, z]:
        c = hermval(arg, c)
    return c


def hermint(c, order=1, k=None, lbnd=0, scl=1, axis=0):
    if k is None:
        k = []
    c = _as_series(c)

    lbnd, scl = map(asarray, (lbnd, scl))

    if not iterable(k):
        k = [k]

    if len(k) > order:
        raise ValueError

    if ndim(lbnd) != 0:
        raise ValueError

    if ndim(scl) != 0:
        raise ValueError

    if order == 0:
        return c

    c = moveaxis(c, axis, 0)
    k = array(list(k) + [0] * (order - len(k)), ndmin=1)

    for i in range(order):
        n = len(c)
        c *= scl
        tmp = empty((n + 1,) + c.shape[1:], dtype=c.dtype)
        tmp = tmp.at[0].set(c[0] * 0)
        tmp = tmp.at[1].set(c[0] / 2)
        j = arange(1, n)
        tmp = tmp.at[j + 1].set(transpose(transpose(c[j]) / (2 * (j + 1))))
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
    input,
    other,
):
    return _subtract(input, other)


def hermval(x, c, tensor=True):
    c = _as_series(c)

    if tensor:
        c = reshape(c, c.shape + (1,) * x.ndim)

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
        raise ValueError

    x = array(x, ndmin=1)
    dims = (degree + 1,) + x.shape
    dtyp = promote_types(x.dtype, array(0.0).dtype)
    x = x.astype(dtyp)
    v = empty(dims, dtype=dtyp)
    v = v.at[0].set(ones_like(x))
    if degree > 0:
        x2 = x * 2
        v = v.at[1].set(x2)

        for i in range(2, degree + 1):
            v = v.at[i].set(v[i - 1] * x2 - v[i - 2] * (2 * (i - 1)))

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
    input,
    other,
):
    return _add(input, other)


def lagcompanion(c):
    c = _as_series(c)

    if len(c) < 2:
        raise ValueError

    if len(c) == 2:
        return array([[1 + c[0] / c[1]]])

    n = len(c) - 1

    mat = reshape(zeros((n, n), dtype=c.dtype))

    mat = mat.at[1 :: n + 1].set(-arange(1, n))
    mat = mat.at[0 :: n + 1].set(2.0 * arange(n) + 1.0)
    mat = mat.at[n :: n + 1].set(-arange(1, n))
    mat = reshape(mat, (n, n))
    mat = mat.at[:, -1].add((c[:-1] / c[-1]) * n)
    return mat


def lagder(c, order=1, scl=1, axis=0):
    if order < 0:
        raise ValueError

    c = _as_series(c)

    if order == 0:
        return c

    c = moveaxis(c, axis, 0)
    n = len(c)
    if order >= n:
        c = zeros_like(c[:1])
    else:
        for _ in range(order):
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
    input,
    other,
):
    return _div(lagmul, input, other)


def lagfit(
    input: Array,
    other: Array,
    degree: Array | int,
    relative_condition: float | None = None,
    full: bool = False,
    weight: Array | None = None,
):
    return _fit(
        lagvander,
        input,
        other,
        degree,
        relative_condition,
        full,
        weight,
    )


def lagfromroots(roots):
    return _from_roots(lagline, lagmul, roots)


def laggauss(degree):
    degree = int(degree)
    if degree <= 0:
        raise ValueError

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
    for arg in [x, y]:
        c = lagval(arg, c)
    return c


def laggrid3d(x, y, z, c):
    for arg in [x, y, z]:
        c = lagval(arg, c)
    return c


def lagint(c, order=1, k=None, lbnd=0, scl=1, axis=0):
    if k is None:
        k = []

    c = _as_series(c)

    lbnd, scl = map(asarray, (lbnd, scl))

    if not iterable(k):
        k = [k]
    if len(k) > order:
        raise ValueError
    if ndim(lbnd) != 0:
        raise ValueError
    if ndim(scl) != 0:
        raise ValueError

    if order == 0:
        return c

    c = moveaxis(c, axis, 0)
    k = array(list(k) + [0] * (order - len(k)), ndmin=1)

    for i in range(order):
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


def lagsub(input, other):
    return _subtract(input, other)


def lagval(x, c, tensor=True):
    c = _as_series(c)

    if tensor:
        c = reshape(c, c.shape + (1,) * x.ndim)

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
        raise ValueError

    x = array(x, ndmin=1)
    dims = (degree + 1,) + x.shape
    dtyp = promote_types(x.dtype, array(0.0).dtype)
    x = x.astype(dtyp)
    v = empty(dims, dtype=dtyp)
    v = v.at[0].set(ones_like(x))
    if degree > 0:
        v = v.at[1].set(1 - x)

        for i in range(2, degree + 1):
            v = v.at[i].set((v[i - 1] * (2 * i - 1 - x) - v[i - 2] * (i - 1)) / i)

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
    input,
    other,
):
    return _add(input, other)


def legcompanion(c):
    c = _as_series(c)
    if len(c) < 2:
        raise ValueError
    if len(c) == 2:
        return array([[-c[0] / c[1]]])

    n = len(c) - 1
    mat = zeros((n, n), dtype=c.dtype)
    scl = 1.0 / sqrt(2 * arange(n) + 1)
    shp = mat.shape
    mat = reshape(mat, [-1])
    mat = mat.at[1 :: n + 1].set(arange(1, n) * scl[: n - 1] * scl[1:n])
    mat = mat.at[n :: n + 1].set(arange(1, n) * scl[: n - 1] * scl[1:n])
    mat = reshape(mat, shp)
    mat = mat.at[:, -1].add(-(c[:-1] / c[-1]) * (scl / scl[-1]) * (n / (2 * n - 1)))
    return mat


def legder(c, order=1, scl=1, axis=0):
    if order < 0:
        raise ValueError

    c = _as_series(c)

    if order == 0:
        return c

    c = moveaxis(c, axis, 0)
    n = len(c)
    if order >= n:
        c = zeros_like(c[:1])
    else:
        for _ in range(order):
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
    input,
    other,
):
    return _div(legmul, input, other)


def legfit(
    input: Array,
    other: Array,
    degree: Array | int,
    relative_condition: float | None = None,
    full: bool = False,
    weight: Array | None = None,
):
    return _fit(
        legvander,
        input,
        other,
        degree,
        relative_condition,
        full,
        weight,
    )


def legfromroots(roots):
    return _from_roots(legline, legmul, roots)


def leggauss(degree):
    degree = int(degree)
    if degree <= 0:
        raise ValueError

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
    for arg in [x, y]:
        c = legval(arg, c)
    return c


def leggrid3d(x, y, z, c):
    for arg in [x, y, z]:
        c = legval(arg, c)
    return c


def legint(c, order=1, k=None, lbnd=0, scl=1, axis=0):
    if k is None:
        k = []

    c = _as_series(c)

    lbnd, scl = map(asarray, (lbnd, scl))

    if not iterable(k):
        k = [k]
    if len(k) > order:
        raise ValueError
    if ndim(lbnd) != 0:
        raise ValueError
    if ndim(scl) != 0:
        raise ValueError

    if order == 0:
        return c

    c = moveaxis(c, axis, 0)
    k = array(list(k) + [0] * (order - len(k)), ndmin=1)

    for i in range(order):
        n = len(c)
        c *= scl
        tmp = empty((n + 1,) + c.shape[1:], dtype=c.dtype)
        tmp = tmp.at[0].set(c[0] * 0)
        tmp = tmp.at[1].set(c[0])
        if n > 1:
            tmp = tmp.at[2].set(c[1] / 3)
        j = arange(2, n)
        t = transpose(transpose(c[j]) / (2 * j + 1))
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

    def body(i, prd):
        j = i + 1

        k = i - 1

        s = i + j

        prd = prd.at[j].set((c[i] * j) / s)

        prd = prd.at[k].add((c[i] * i) / s)

        return prd

    b = len(c)
    prd = zeros(len(c) + 1, dtype=c.dtype).at[1].set(c[0])

    for index in range(1, b):
        prd = body(index, prd)

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
    input,
    other,
):
    return _subtract(input, other)


def legval(x, c, tensor=True):
    c = _as_series(c)

    if tensor:
        c = reshape(c, c.shape + (1,) * x.ndim)

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
        raise ValueError

    x = array(x, ndmin=1)

    dims = (degree + 1,) + x.shape

    dtyp = promote_types(x.dtype, array(0.0).dtype)

    x = x.astype(dtyp)

    v = empty(dims, dtype=dtyp)

    v = v.at[0].set(ones_like(x))

    if degree > 0:
        v = v.at[1].set(x)

        for i in range(2, degree + 1):
            v = v.at[i].set((v[i - 1] * x * (2 * i - 1) - v[i - 2] * (i - 1)) / i)

    return moveaxis(v, 0, -1)


def legvander2d(x, y, degree):
    return _vander_nd_flat(
        (legvander, legvander),
        (x, y),
        degree,
    )


def legvander3d(x, y, z, degree):
    return _vander_nd_flat(
        (legvander, legvander, legvander),
        (x, y, z),
        degree,
    )


def legweight(x):
    return ones_like(x)


def _map_domain(x, old, new):
    oldlen = old[1] - old[0]
    newlen = new[1] - new[0]
    off1 = (old[1] * new[0] - old[0] * new[1]) / oldlen
    scl1 = newlen / oldlen
    off, scl = off1, scl1

    return off + scl * x


def _map_parameters(previous, new):
    oldlen = previous[1] - previous[0]

    newlen = new[1] - new[0]

    off = (previous[1] * new[0] - previous[0] * new[1]) / oldlen

    scl = newlen / oldlen

    return off, scl


def poly2cheb(input):
    input = _as_series(input)

    y = zeros_like(input)

    for i in range(0, input.shape[0] - 1 + 1):
        y = chebadd(
            chebmulx(y, mode="same"),
            input[input.shape[0] - 1 - i],
        )

    return y


def poly2herm(input):
    input = _as_series(input)

    output = zeros_like(input)

    for i in range(0, input.shape[0] - 1 + 1):
        output = hermadd(
            hermmulx(output, mode="same"),
            input[input.shape[0] - 1 - i],
        )

    return output


def poly2herme(input):
    input = _as_series(input)

    degree = input.shape[0] - 1

    output = zeros_like(input)

    for i in range(0, degree + 1):
        output = hermeadd(
            hermemulx(output, mode="same"),
            input[degree - i],
        )

    return output


def poly2lag(input):
    input = _as_series(input)

    output = zeros_like(input)

    for i in range(0, input.shape[0]):
        output = lagadd(
            lagmulx(output, mode="same"),
            input[::-1][i],
        )

    return output


def poly2leg(input):
    input = _as_series(input)

    output = zeros_like(input)

    for i in range(0, input.shape[0] - 1 + 1):
        output = legadd(
            legmulx(output, mode="same"),
            input[input.shape[0] - 1 - i],
        )

    return output


def polyadd(input, other):
    return _add(input, other)


def polycompanion(c):
    c = _as_series(c)

    if len(c) < 2:
        raise ValueError

    if len(c) == 2:
        return array([[-c[0] / c[1]]])

    n = len(c) - 1

    mat = reshape(zeros((n, n), dtype=c.dtype), [-1])

    mat = mat.at[n :: n + 1].set(1)

    mat = reshape(mat, (n, n))

    return mat.at[:, -1].add(-c[:-1] / c[-1])


def polyder(c, order=1, scl=1, axis=0):
    c = _as_series(c)

    if order == 0:
        return c

    c = moveaxis(c, axis, 0)

    n = c.shape[0]

    if order >= n:
        c = zeros_like(c[:1])
    else:
        d = arange(n)

        def body(_, c):
            c = transpose(d * transpose(c))

            c = roll(c, -1, axis=0) * scl

            c = c.at[-1].set(0)

            return c

        y = c

        for index in range(0, order):
            y = body(index, y)

        c = y

        c = c[:-order]

    c = moveaxis(c, 0, axis)

    return c


def polydiv(input, other):
    input, other = _as_series(input, other)

    return _div(polymul, input, other)


def polyfit(
    input: Array,
    other: Array,
    degree: Array | int,
    relative_condition: float | None = None,
    full: bool = False,
    weight: Array | None = None,
):
    return _fit(
        polyvander,
        input,
        other,
        degree,
        relative_condition,
        full,
        weight,
    )


def polyfromroots(roots):
    return _from_roots(polyline, polymul, roots)


def polygrid2d(x, y, c):
    for arg in [x, y]:
        c = polyval(arg, c)
    return c


def polygrid3d(x, y, z, c):
    for arg in [x, y, z]:
        c = polyval(arg, c)
    return c


def polyint(c, order=1, k=None, lbnd=0, scl=1, axis=0):
    if k is None:
        k = []

    c = _as_series(c)

    lbnd, scl = map(asarray, (lbnd, scl))

    if not iterable(k):
        k = [k]

    if order < 0:
        raise ValueError

    if len(k) > order:
        raise ValueError

    if ndim(lbnd) != 0:
        raise ValueError

    if ndim(scl) != 0:
        raise ValueError

    if order == 0:
        return c

    k = array(list(k) + [0] * (order - len(k)), ndmin=1)

    n = c.shape[axis]

    c = _pad_along_axis(c, (0, order), axis)

    c = moveaxis(c, axis, 0)

    d = arange(n + order) + 1

    for i in range(0, order):
        c = c * scl

        c = transpose(transpose(c) / d)

        c = roll(c, 1, axis=0)

        c = c.at[0].set(0)

        c = c.at[0].add(k[i] - polyval(lbnd, c))

    return moveaxis(c, 0, axis)


def polyline(off, scl):
    return array([off, scl])


def polymul(input, other, mode="full"):
    input, other = _as_series(input, other)

    output = convolve(input, other)

    if mode == "same":
        output = output[: max(len(input), len(other))]

    return output


def polymulx(input, mode="full"):
    input = _as_series(input)

    output = zeros(len(input) + 1, dtype=input.dtype)

    output = output.at[1:].set(input)

    if mode == "same":
        output = output[: len(input)]

    return output


def polypow(c, pow, maxpower=16):
    return _pow(polymul, c, pow, maxpower)


def polyroots(input):
    input = _as_series(input)

    if len(input) < 2:
        return array([], dtype=input.dtype)

    if len(input) == 2:
        return array([-input[0] / input[1]])

    return sort(eigvals(polycompanion(input)[::-1, ::-1]))


def polysub(input, other):
    return _subtract(input, other)


def polyval(x, c, tensor=True):
    c = _as_series(c)

    if tensor:
        c = reshape(c, c.shape + (1,) * x.ndim)

    c0 = c[-1] + zeros_like(x)

    y = c0

    for index in range(2, c.shape[0] + 1):
        y = c[-index] + y * x

    return y


def polyval2d(x, y, c):
    return _evaluate(
        polyval,
        c,
        x,
        y,
    )


def polyval3d(x, y, z, c):
    return _evaluate(
        polyval,
        c,
        x,
        y,
        z,
    )


def polyvalfromroots(x, r, tensor=True):
    r = array(r, ndmin=1)

    if tensor:
        r = reshape(r, r.shape + (1,) * x.ndim)

    if x.ndim >= r.ndim:
        raise ValueError

    output = prod(x - r, axis=0)

    return output


def polyvander(input, degree):
    if degree < 0:
        raise ValueError

    input = array(input, ndmin=1)

    output = empty((degree + 1,) + input.shape, dtype=input.dtype)

    output = output.at[0].set(ones_like(input))

    for index in range(1, degree + 1):
        output = output.at[index].set(output[index - 1] * input)

    output = moveaxis(output, 0, -1)

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
        raise ValueError

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
    "_as_series",
    "_map_domain",
    "_map_parameters",
    "_trim_coefficients",
    "_trim_sequence",
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
    "_get_domain",
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
