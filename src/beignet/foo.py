import functools
import math
import operator
from typing import Callable, Literal, Tuple

import jax
import jax.numpy
from jax.numpy import (
    abs,
    arange,
    array,
    asarray,
    complexfloating,
    convolve,
    finfo,
    flip,
    full,
    iscomplexobj,
    moveaxis,
    nonzero,
    ones,
    ones_like,
    pad,
    ravel,
    reshape,
    roll,
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
    lstsq,
)
from torch import Tensor

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


def _add(input: Tensor, other: Tensor) -> Tensor:
    input, other = _as_series(input, other)

    if input.shape[0] > other.shape[0]:
        output = input.at[: math.prod(other.shape)].add(other)
    else:
        output = other.at[: math.prod(input.shape)].add(input)

    return output


def _as_series(*args, trim: bool = False) -> Tuple[Tensor, ...]:
    xs = ()

    for arg in args:
        x = array(arg)

        if x.ndim == 0:
            x = ravel(x)

        if trim:
            x = _trim_sequence(x)

        xs = *xs, x

    xs = jax._src.numpy.util.promote_dtypes_inexact(*xs)

    if len(xs) == 1:
        return xs[0]

    return xs


def _c_series_to_z_series(input: Tensor) -> Tensor:
    n = math.prod(input.shape)

    zs = zeros(2 * n - 1, dtype=input.dtype)

    zs = zs.at[n - 1 :].set(input / 2)

    output = flip(zs, axis=0)

    output = output + zs

    return output


def _div(func: Callable, input: Tensor, other: Tensor) -> Tuple[Tensor, Tensor]:
    input, other = _as_series(input, other)

    lc1 = input.shape[0]
    lc2 = other.shape[0]

    if lc1 < lc2:
        return zeros_like(input[:1]), input

    if lc2 == 1:
        return input / other[-1], zeros_like(input[:1])

    def _ldordidx(x):
        indicies = nonzero(flip(x, axis=0), size=1)

        return x.shape[0] - 1 - indicies[0][0]

    quotient = zeros(lc1 - lc2 + 1, dtype=input.dtype)

    ridx = input.shape[0] - 1

    sz = lc1 - _ldordidx(other) - 1

    y = zeros(lc1 + lc2 + 1, dtype=input.dtype)
    y = y.at[sz].set(1.0)

    y1 = quotient, input, y, ridx

    for index in range(0, sz):
        quotient, remainder, y2, ridx1 = y1

        i = sz - index

        p = func(y2, other)

        pidx = _ldordidx(p)

        t = remainder[ridx1] / p[pidx]

        a = remainder.at[ridx1].set(0)
        b = t * p.at[pidx].set(0)

        remainder = _subtract(a, b)
        remainder = remainder[: remainder.shape[0]]

        quotient = quotient.at[i].set(t)

        ridx1 = ridx1 - 1

        y2 = roll(y2, -1)

        y1 = quotient, remainder, y2, ridx1

    quotient, remainder, _, _ = y1

    return quotient, remainder


def _evaluate(func: Callable, input: Tensor, *args) -> Tensor:
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


def _fit(
    vandermonde_func,
    input: Tensor,
    other: Tensor,
    degree: Tensor | int,
    relative_condition: float | None = None,
    full: bool = False,
    weight: Tensor | None = None,
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

    a = transpose(vandermonde)
    b = transpose(other)

    if weight is not None:
        weight = asarray(weight)

        if weight.ndim != 1:
            raise TypeError

        if input.shape[0] != weight.shape[0]:
            raise TypeError

        a = a * weight
        b = b * weight

    if relative_condition is None:
        relative_condition = input.shape[0] * finfo(input.dtype).eps

    if issubclass(a.dtype.type, complexfloating):
        scale = square(a.real) + square(a.imag)

        scale = sum(
            scale,
            axis=1,
        )

        scale = sqrt(scale)
    else:
        scale = square(a)

        scale = sum(
            scale,
            axis=1,
        )

        scale = sqrt(scale)

    scale = where(scale == 0, 1, scale)

    output, residuals, rank, s = lstsq(
        transpose(a) / scale,
        transpose(b),
        relative_condition,
    )

    output = transpose(transpose(output) / scale)

    if degree.ndim > 0:
        if output.ndim == 2:
            x = zeros((maximum + 1, output.shape[1]), dtype=output.dtype)

            output = x.at[degree].set(output)
        else:
            x = zeros(maximum + 1, dtype=output.dtype)

            output = x.at[degree].set(output)

    if full:
        return output, [residuals, rank, s, relative_condition]
    else:
        return output


def _flattened_vandermonde(vander_fs, points, degrees):
    v = _vandermonde(vander_fs, points, degrees)
    return reshape(v, v.shape[: -len(degrees)] + (-1,))


def _from_roots(f: Callable, g: Callable, input: Tensor) -> Tensor:
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


def _get_domain(x: Tensor) -> Tensor:
    if iscomplexobj(x):
        rmin, rmax = x.real.min(), x.real.max()
        imin, imax = x.imag.min(), x.imag.max()

        return array(((rmin + 1j * imin), (rmax + 1j * imax)))

    return array((x.min(), x.max()))


def _map_domain(x, old, new):
    oldlen = old[1] - old[0]
    newlen = new[1] - new[0]
    off1 = (old[1] * new[0] - old[0] * new[1]) / oldlen
    scl1 = newlen / oldlen
    off, scale = off1, scl1

    return off + scale * x


def _map_parameters(input: Tensor, other: Tensor) -> Tuple[Tensor, Tensor]:
    a = input[1] - input[0]
    b = other[1] - other[0]

    x = (input[1] * other[0] - input[0] * other[1]) / a
    y = b / a

    return x, y


def _normed_hermite_e_n(x: Tensor, n):
    if n == 0:
        output = full(x.shape, 1 / sqrt(sqrt(2 * math.pi)))
    else:
        a = zeros_like(x)
        b = ones_like(x) / sqrt(sqrt(2 * math.pi))

        d = array(n)

        for _ in range(0, n - 1):
            c = a

            a = -b * sqrt((d - 1.0) / d)

            b = c + b * x * sqrt(1.0 / d)

            d = d - 1.0

        output = a + b * x

    return output


def _normed_hermite_n(x: Tensor, n):
    if n == 0:
        output = full(x.shape, 1 / sqrt(sqrt(math.pi)))
    else:
        a = zeros_like(x)

        b = ones_like(x) / sqrt(sqrt(math.pi))

        d = array(n)

        for _ in range(0, n - 1):
            c = a

            a = -b * sqrt((d - 1.0) / d)

            b = c + b * x * sqrt(2.0 / d)

            d = d - 1.0

        output = a + b * x * math.sqrt(2)

    return output


def _nth_slice(i, ndim):
    sl = [None] * ndim
    sl[i] = slice(None)
    return tuple(sl)


def _pad_along_axis(input: Tensor, padding=(0, 0), axis=0):
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


def _pow(func: Callable, input: Tensor, exponent, maximum_exponent) -> Tensor:
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

    output = zeros(input.shape[0] * exponent, dtype=input.dtype)

    output = _add(output, input)

    for _ in range(2, power + 1):
        output = func(output, input, mode="same")

    return output


def _subtract(input: Tensor, other: Tensor) -> Tensor:
    input, other = _as_series(input, other)

    if input.shape[0] > other.shape[0]:
        output = -other

        output = input.at[: math.prod(other.shape)].add(output)

        return output

    output = -other

    output = output.at[: math.prod(input.shape)].add(input)

    return output


def _trim_coefficients(input, tol=0):
    if tol < 0:
        raise ValueError

    input = _as_series(input)

    [ind] = nonzero(abs(input) > tol)

    if len(ind) == 0:
        return input[:1] * 0
    else:
        return input[: ind[-1] + 1]


def _trim_sequence(seq):
    if len(seq) == 0:
        return seq
    else:
        for i in range(len(seq) - 1, -1, -1):
            if seq[i] != 0:
                break

        return seq[: i + 1]


def _vandermonde(vander_fs, points, degrees):
    n_dims = len(vander_fs)

    if n_dims != len(points):
        raise ValueError

    if n_dims != len(degrees):
        raise ValueError

    if n_dims == 0:
        raise ValueError

    points = tuple(array(tuple(points)) + 0.0)

    output = []

    for i in range(n_dims):
        vandermonde = vander_fs[i](points[i], degrees[i])

        vandermonde = vandermonde[(..., *_nth_slice(i, n_dims))]

        output = [*output, vandermonde]

    return functools.reduce(operator.mul, output)


def _z_series_mul(input: Tensor, other: Tensor, mode="full") -> Tensor:
    return convolve(input, other, mode=mode)


def _z_series_to_c_series(input: Tensor) -> Tensor:
    n = (math.prod(input.shape) + 1) // 2
    c = input[n - 1 :]
    return c.at[1:n].multiply(2)


def chebadd(input: Tensor, other: Tensor) -> Tensor:
    return _add(input, other)


def chebdiv(input: Tensor, other: Tensor) -> Tuple[Tensor, Tensor]:
    return _div(chebmul, input, other)


def chebline(off, scale):
    return array([off, scale])


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
        output = output[: max(input.shape[0], other.shape[0])]

    return output


def chebmulx(input: Tensor, mode="full") -> Tensor:
    input = _as_series(input)

    output = zeros(input.shape[0] + 1, dtype=input.dtype)

    output = output.at[1].set(input[0])

    if input.shape[0] > 1:
        tmp = input[1:] / 2

        output = output.at[2:].set(tmp)

        output = output.at[0:-2].add(tmp)

    if mode == "same":
        output = output[: input.shape[0]]

    return output


def chebpow(
    input: Tensor,
    exponent: int,
    maximum_exponent: int = 16,
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

    output = zeros(input.shape[0] * exponent, dtype=input.dtype)

    output = chebadd(output, input)

    zs = _c_series_to_z_series(input)

    output = _c_series_to_z_series(output)

    y = output

    for _ in range(2, power + 1):
        y = convolve(y, zs, mode="same")

    output = y

    output = _z_series_to_c_series(output)

    return output


def chebsub(input: Tensor, other: Tensor) -> Tensor:
    return _subtract(input, other)


def hermadd(input: Tensor, other: Tensor) -> Tensor:
    return _add(input, other)


def hermdiv(input: Tensor, other: Tensor) -> Tuple[Tensor, Tensor]:
    return _div(hermmul, input, other)


def hermeadd(input: Tensor, other: Tensor) -> Tensor:
    return _add(input, other)


def hermediv(input: Tensor, other: Tensor) -> Tuple[Tensor, Tensor]:
    return _div(hermemul, input, other)


def hermeline(input: float, other: float) -> Tensor:
    return array([input, other])


def hermemul(input, other, mode="full"):
    input, other = _as_series(input, other)
    lc1, lc2 = input.shape[0], other.shape[0]
    if lc1 > lc2:
        c = other
        xs = input
    else:
        c = input
        xs = other

    if c.shape[0] == 1:
        c0 = hermeadd(zeros(lc1 + lc2 - 1), c[0] * xs)
        input = zeros(lc1 + lc2 - 1)
    elif c.shape[0] == 2:
        c0 = hermeadd(zeros(lc1 + lc2 - 1), c[0] * xs)
        input = hermeadd(zeros(lc1 + lc2 - 1), c[1] * xs)
    else:
        nd = c.shape[0]
        c0 = hermeadd(zeros(lc1 + lc2 - 1), c[-2] * xs)
        input = hermeadd(zeros(lc1 + lc2 - 1), c[-1] * xs)

        def body(i, val):
            c0, c1, nd = val
            tmp = c0
            nd = nd - 1
            c0 = hermesub(c[-i] * xs, c1 * (nd - 1))
            c1 = hermeadd(tmp, hermemulx(c1, "same"))
            return c0, c1, nd

        b = c.shape[0] + 1
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

    output = zeros(c.shape[0] + 1, dtype=c.dtype)

    output = output.at[1].set(c[0])

    i = arange(1, c.shape[0])

    output = output.at[i + 1].set(c[i])
    output = output.at[i - 1].add(c[i] * i)

    if mode == "same":
        output = output[: c.shape[0]]

    return output


def hermepow(c, exponent, maximum_exponent=16):
    return _pow(hermemul, c, exponent, maximum_exponent)


def hermesub(input: Tensor, other: Tensor) -> Tensor:
    return _subtract(input, other)


def hermline(off, scale):
    return array([off, scale / 2])


def hermmul(input, other, mode="full"):
    input, other = _as_series(input, other)
    lc1, lc2 = input.shape[0], other.shape[0]
    if lc1 > lc2:
        c = other
        xs = input
    else:
        c = input
        xs = other

    if c.shape[0] == 1:
        c0 = hermadd(zeros(lc1 + lc2 - 1), c[0] * xs)
        input = zeros(lc1 + lc2 - 1)
    elif c.shape[0] == 2:
        c0 = hermadd(zeros(lc1 + lc2 - 1), c[0] * xs)
        input = hermadd(zeros(lc1 + lc2 - 1), c[1] * xs)
    else:
        nd = c.shape[0]
        c0 = hermadd(zeros(lc1 + lc2 - 1), c[-2] * xs)
        input = hermadd(zeros(lc1 + lc2 - 1), c[-1] * xs)

        def body(i, val):
            c0, c1, nd = val
            tmp = c0
            nd = nd - 1
            c0 = hermsub(c[-i] * xs, c1 * (2 * (nd - 1)))
            c1 = hermadd(tmp, hermmulx(c1, "same") * 2)
            return c0, c1, nd

        b = c.shape[0] + 1
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
    output = zeros(c.shape[0] + 1, dtype=c.dtype)
    output = output.at[1].set(c[0] / 2)

    i = arange(1, c.shape[0])

    output = output.at[i + 1].set(c[i] / 2)
    output = output.at[i - 1].add(c[i] * i)

    if mode == "same":
        output = output[: c.shape[0]]

    return output


def hermpow(c, exponent, maximum_exponent=16):
    return _pow(hermmul, c, exponent, maximum_exponent)


def hermsub(input, other):
    return _subtract(input, other)


def lagadd(input: Tensor, other: Tensor) -> Tensor:
    return _add(input, other)


def lagdiv(input, other):
    return _div(lagmul, input, other)


def lagline(off, scale):
    return array([off + scale, -scale])


def lagmul(input, other, mode="full"):
    input, other = _as_series(input, other)
    lc1, lc2 = input.shape[0], other.shape[0]

    if lc1 > lc2:
        c = other
        xs = input
    else:
        c = input
        xs = other

    if c.shape[0] == 1:
        c0 = lagadd(zeros(lc1 + lc2 - 1), c[0] * xs)
        input = zeros(lc1 + lc2 - 1)
    elif c.shape[0] == 2:
        c0 = lagadd(zeros(lc1 + lc2 - 1), c[0] * xs)
        input = lagadd(zeros(lc1 + lc2 - 1), c[1] * xs)
    else:
        nd = c.shape[0]
        c0 = lagadd(zeros(lc1 + lc2 - 1), c[-2] * xs)
        input = lagadd(zeros(lc1 + lc2 - 1), c[-1] * xs)

        def body(i, val):
            c0, c1, nd = val
            tmp = c0
            nd = nd - 1
            c0 = lagsub(c[-i] * xs, (c1 * (nd - 1)) / nd)
            c1 = lagadd(tmp, lagsub((2 * nd - 1) * c1, lagmulx(c1, "same")) / nd)
            return c0, c1, nd

        b = c.shape[0] + 1
        x = (c0, input, nd)
        y = x
        for index in range(3, b):
            y = body(index, y)
        c0, input, _ = y

    ret = lagadd(c0, lagsub(input, lagmulx(input, "same")))
    if mode == "same":
        ret = ret[: max(lc1, lc2)]
    return ret


def lagmulx(input, mode="full"):
    input = _as_series(input)

    output = zeros(input.shape[0] + 1, dtype=input.dtype)
    output = output.at[0].set(input[0])
    output = output.at[1].set(-input[0])

    i = arange(1, input.shape[0])

    output = output.at[i + 1].set(-input[i] * (i + 1))
    output = output.at[i].add(input[i] * (2 * i + 1))
    output = output.at[i - 1].add(-input[i] * i)

    if mode == "same":
        output = output[: input.shape[0]]
    return output


def lagpow(
    input: Tensor,
    exponent: float,
    maximum_exponent: float = 16,
) -> Tensor:
    return _pow(lagmul, input, exponent, maximum_exponent)


def lagsub(input: Tensor, other: Tensor) -> Tensor:
    return _subtract(input, other)


def legadd(input: Tensor, other: Tensor) -> Tensor:
    return _add(input, other)


def legdiv(input: Tensor, other: Tensor) -> Tensor:
    return _div(legmul, input, other)


def legline(input: float, other: float) -> Tensor:
    return array([input, other])


def legmul(input, other, mode="full"):
    input, other = _as_series(input, other)
    lc1, lc2 = input.shape[0], other.shape[0]
    if lc1 > lc2:
        c = other
        xs = input
    else:
        c = input
        xs = other

    if c.shape[0] == 1:
        c0 = legadd(zeros(lc1 + lc2 - 1), c[0] * xs)
        input = zeros(lc1 + lc2 - 1)
    elif c.shape[0] == 2:
        c0 = legadd(zeros(lc1 + lc2 - 1), c[0] * xs)
        input = legadd(zeros(lc1 + lc2 - 1), c[1] * xs)
    else:
        nd = c.shape[0]
        c0 = legadd(zeros(lc1 + lc2 - 1), c[-2] * xs)
        input = legadd(zeros(lc1 + lc2 - 1), c[-1] * xs)

        def body(i, val):
            c0, c1, nd = val
            tmp = c0
            nd = nd - 1
            c0 = legsub(c[-i] * xs, (c1 * (nd - 1)) / nd)
            c1 = legadd(tmp, (legmulx(c1, "same") * (2 * nd - 1)) / nd)
            return c0, c1, nd

        b = c.shape[0] + 1
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

    def body(i, output):
        j = i + 1

        k = i - 1

        s = i + j

        output = output.at[j].set((c[i] * j) / s)

        output = output.at[k].add((c[i] * i) / s)

        return output

    b = c.shape[0]

    output = zeros(c.shape[0] + 1, dtype=c.dtype).at[1].set(c[0])

    for i in range(1, b):
        output = body(i, output)

    if mode == "same":
        output = output[: c.shape[0]]

    return output


def legpow(c, exponent, maximum_exponent=16):
    return _pow(legmul, c, exponent, maximum_exponent)


def legsub(input: Tensor, other: Tensor) -> Tensor:
    return _subtract(input, other)


def polyadd(input: Tensor, other: Tensor) -> Tensor:
    return _add(input, other)


def polydiv(input: Tensor, other: Tensor) -> Tuple[Tensor, Tensor]:
    input, other = _as_series(input, other)

    return _div(polymul, input, other)


def polyline(input: float, other: float) -> Tensor:
    return array([input, other])


def polymul(input: Tensor, other: Tensor, mode="full") -> Tensor:
    input, other = _as_series(input, other)

    output = convolve(input, other)

    if mode == "same":
        output = output[: max(input.shape[0], other.shape[0])]

    return output


def polymulx(input: Tensor, mode="full") -> Tensor:
    input = _as_series(input)

    output = zeros(input.shape[0] + 1, dtype=input.dtype)

    output = output.at[1:].set(input)

    if mode == "same":
        output = output[: input.shape[0]]

    return output


def polypow(
    input: Tensor,
    exponent: float,
    maximum_exponent: float = 16,
) -> Tensor:
    return _pow(
        polymul,
        input,
        exponent,
        maximum_exponent,
    )


def polysub(input: Tensor, other: Tensor) -> Tensor:
    return _subtract(input, other)


chebtrim = _trim_coefficients
hermetrim = _trim_coefficients
hermtrim = _trim_coefficients
lagtrim = _trim_coefficients
legtrim = _trim_coefficients
polytrim = _trim_coefficients

__all__ = [
    "_as_series",
    "_get_domain",
    "_map_domain",
    "_map_parameters",
    "_trim_coefficients",
    "_trim_sequence",
    "chebadd",
    "chebdiv",
    "chebdomain",
    "chebline",
    "chebmul",
    "chebmulx",
    "chebone",
    "chebpow",
    "chebsub",
    "chebtrim",
    "chebx",
    "chebzero",
    "hermadd",
    "hermdiv",
    "hermdomain",
    "hermeadd",
    "hermediv",
    "hermedomain",
    "hermeline",
    "hermemul",
    "hermemulx",
    "hermeone",
    "hermepow",
    "hermesub",
    "hermetrim",
    "hermex",
    "hermezero",
    "hermline",
    "hermmul",
    "hermmulx",
    "hermone",
    "hermpow",
    "hermsub",
    "hermtrim",
    "hermx",
    "hermzero",
    "lagadd",
    "lagdiv",
    "lagdomain",
    "lagline",
    "lagmul",
    "lagmulx",
    "lagone",
    "lagpow",
    "lagsub",
    "lagtrim",
    "lagx",
    "lagzero",
    "legadd",
    "legdiv",
    "legdomain",
    "legline",
    "legmul",
    "legmulx",
    "legone",
    "legpow",
    "legsub",
    "legtrim",
    "legx",
    "legzero",
    "polyadd",
    "polydiv",
    "polydomain",
    "polyline",
    "polymul",
    "polymulx",
    "polyone",
    "polypow",
    "polysub",
    "polytrim",
    "polyx",
    "polyzero",
]
