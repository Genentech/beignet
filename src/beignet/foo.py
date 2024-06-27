import functools
import math
import operator
from typing import Callable, Literal, Tuple

import jax
import jax.numpy
from jax import Array
from jax.numpy import (
    abs,
    any,
    arange,
    array,
    asarray,
    complexfloating,
    convolve,
    finfo,
    flip,
    full,
    imag,
    moveaxis,
    ndim,
    nonzero,
    ones,
    ones_like,
    pad,
    ravel,
    real,
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


def _add(input: Array, other: Array) -> Array:
    input, other = _as_series(input, other)

    if input.shape[0] > other.shape[0]:
        output = input.at[: math.prod(other.shape)].add(other)
    else:
        output = other.at[: math.prod(input.shape)].add(input)

    return output


def _as_series(*args, trim: bool = False) -> Tuple[Array, ...]:
    xs = ()

    for arg in args:
        x = array(arg)

        if ndim(x) == 0:
            x = ravel(x)

        if trim:
            x = _trim_sequence(x)

        xs = *xs, x

    xs = jax._src.numpy.util.promote_dtypes_inexact(*xs)

    if len(xs) == 1:
        return xs[0]

    return xs


def _c_series_to_z_series(input: Array) -> Array:
    n = math.prod(input.shape)

    zs = zeros(2 * n - 1, dtype=input.dtype)

    zs = zs.at[n - 1 :].set(input / 2.0)

    return flip(zs, axis=0) + zs


def _div(func: Callable, input: Array, other: Array) -> Tuple[Array, Array]:
    input, other = _as_series(input, other)

    m = input.shape[0]
    n = other.shape[0]

    if m < n:
        return zeros_like(input[:1]), input

    if n == 1:
        return input / other[-1], zeros_like(input[:1])

    def _ldordidx(x):
        indicies = nonzero(flip(x, axis=0), size=1)

        return x.shape[0] - 1 - indicies[0][0]

    quotient = zeros(m - n + 1, dtype=input.dtype)

    ridx = input.shape[0] - 1

    sz = m - _ldordidx(other) - 1

    y = zeros(m + n + 1, dtype=input.dtype)
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


def _evaluate(func: Callable, input: Array, *args) -> Array:
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
    input: Array,
    other: Array,
    degree: Array | int,
    relative_condition: float | None = None,
    full: bool = False,
    weight: Array | None = None,
):
    degree = asarray(degree)

    if (
        ndim(degree) > 1
        or degree.dtype.kind not in "iu"
        or math.prod(degree.shape) == 0
    ):
        raise TypeError

    if degree.min() < 0:
        raise ValueError

    if ndim(input) != 1:
        raise TypeError

    if math.prod(input.shape) == 0:
        raise TypeError

    if ndim(other) < 1 or ndim(other) > 2:
        raise TypeError

    if input.shape[0] != other.shape[0]:
        raise TypeError

    if ndim(degree) == 0:
        maximum = int(degree)

        v = vandermonde_func(input, maximum)
    else:
        degree = sort(degree)

        maximum = int(degree[-1])

        v = vandermonde_func(input, maximum)[:, degree]

    a = transpose(v)
    b = transpose(other)

    if weight is not None:
        weight = asarray(weight)

        if ndim(weight) != 1:
            raise TypeError

        if input.shape[0] != weight.shape[0]:
            raise TypeError

        a = a * weight
        b = b * weight

    if relative_condition is None:
        relative_condition = input.shape[0] * finfo(input.dtype).eps

    if issubclass(a.dtype.type, complexfloating):
        scale = square(real(a)) + square(imag(a))

        scale = sum(scale, axis=1)

        scale = sqrt(scale)
    else:
        scale = square(a)

        scale = sum(scale, axis=1)

        scale = sqrt(scale)

    scale = where(scale == 0, 1, scale)

    output, residuals, rank, s = lstsq(
        transpose(a) / scale,
        transpose(b),
        relative_condition,
    )

    output = transpose(transpose(output) / scale)

    if ndim(degree) > 0:
        if ndim(output) == 2:
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


def _from_roots(f: Callable, g: Callable, input: Array) -> Array:
    if math.prod(input.shape) == 0:
        return ones([1])

    input = sort(input)

    ys = []

    for x in input:
        y = _add(zeros(input.shape[0] + 1, dtype=x.dtype), f(-x, 1))

        ys = [*ys, y]

    p = stack(ys)

    m = p.shape[0]

    x = m, p

    while x[0] > 1:
        m, r = divmod(x[0], 2)

        z = x[1]

        previous = array([zeros(input.shape[0] + 1, dtype=p.dtype)] * len(p))

        y = previous

        for i in range(0, m):
            y = y.at[i].set(g(z[i], z[i + m])[: input.shape[0] + 1])

        previous = y

        if r:
            previous = previous.at[0].set(
                g(previous[0], z[2 * m])[: input.shape[0] + 1]
            )

        x = m, previous

    _, output = x

    return output[0]


def _get_domain(x: Array) -> Array:
    if any(jax.numpy.iscomplex(x)):
        rmin, rmax = real(x).min(), real(x).max()
        imin, imax = imag(x).min(), imag(x).max()

        return array(((rmin + 1.0j * imin), (rmax + 1.0j * imax)))

    return array((x.min(), x.max()))


def _map_domain(x: Array, y: Array, z: Array) -> Array:
    (a, b), (c, d) = y, z

    return (b * c - a * d) / (b - a) + (d - c) / (b - a) * x


def _map_parameters(input: Array, other: Array) -> Tuple[Array, Array]:
    a = input[1] - input[0]
    b = other[1] - other[0]

    x = (input[1] * other[0] - input[0] * other[1]) / a
    y = b / a

    return x, y


def _normed_hermite_e_n(x: Array, n) -> Array:
    if n == 0:
        output = full(x.shape, 1.0 / sqrt(sqrt(2.0 * math.pi)))
    else:
        a = zeros_like(x)
        b = ones_like(x) / sqrt(sqrt(2.0 * math.pi))

        size = array(n)

        for _ in range(0, n - 1):
            previous = a

            a = -b * sqrt((size - 1.0) / size)

            b = previous + b * x * sqrt(1.0 / size)

            size = size - 1.0

        output = a + b * x

    return output


def _normed_hermite_n(x: Array, n) -> Array:
    if n == 0:
        output = full(x.shape, 1 / sqrt(sqrt(math.pi)))
    else:
        a = zeros_like(x)

        b = ones_like(x) / sqrt(sqrt(math.pi))

        size = array(n)

        for _ in range(0, n - 1):
            previous = a

            a = -b * sqrt((size - 1.0) / size)

            b = previous + b * x * sqrt(2.0 / size)

            size = size - 1.0

        output = a + b * x * math.sqrt(2.0)

    return output


def _nth_slice(i, ndim):
    sl = [None] * ndim
    sl[i] = slice(None)
    return tuple(sl)


def _pad_along_axis(input: Array, padding=(0, 0), axis=0):
    input = moveaxis(input, axis, 0)

    if padding[0] < 0:
        input = input[abs(padding[0]) :]
        padding = (0, padding[1])
    if padding[1] < 0:
        input = input[: -abs(padding[1])]
        padding = (padding[0], 0)

    npad = [(0, 0)] * ndim(input)

    npad[0] = padding

    output = pad(input, pad_width=npad, mode="constant", constant_values=0)

    return moveaxis(output, 0, axis)


def _pow(
    func: Callable,
    input: Array,
    exponent: float | Array,
    maximum_exponent: float | Array,
) -> Array:
    input = _as_series(input)

    _exponent = int(exponent)

    if _exponent != exponent or _exponent < 0:
        raise ValueError

    if maximum_exponent is not None and _exponent > maximum_exponent:
        raise ValueError

    if _exponent == 0:
        return array([1], dtype=input.dtype)

    if _exponent == 1:
        return input

    output = zeros(input.shape[0] * exponent, dtype=input.dtype)

    output = _add(output, input)

    for _ in range(2, _exponent + 1):
        output = func(output, input, mode="same")

    return output


def _subtract(input: Array, other: Array) -> Array:
    input, other = _as_series(input, other)

    if input.shape[0] > other.shape[0]:
        output = -other

        output = input.at[: math.prod(other.shape)].add(output)

        return output

    output = -other

    output = output.at[: math.prod(input.shape)].add(input)

    return output


def _trim_coefficients(input: Array, tol=0):
    if tol < 0:
        raise ValueError

    input = _as_series(input)

    [ind] = nonzero(abs(input) > tol)

    if len(ind) == 0:
        return input[:1] * 0
    else:
        return input[: ind[-1] + 1]


def _trim_sequence(sequence):
    if len(sequence) == 0:
        return sequence
    else:
        for i in range(len(sequence) - 1, -1, -1):
            if sequence[i] != 0:
                break

        return sequence[: i + 1]


def _vandermonde(vander_fs, points, degrees) -> Array:
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


def _z_series_mul(
    input: Array,
    other: Array,
    mode: Literal["full", "same", "valid"] = "full",
) -> Array:
    return convolve(input, other, mode=mode)


def _z_series_to_c_series(input: Array) -> Array:
    n = (math.prod(input.shape) + 1) // 2
    c = input[n - 1 :]
    return c.at[1:n].multiply(2)


def chebadd(input: Array, other: Array) -> Array:
    return _add(input, other)


def chebdiv(input: Array, other: Array) -> Tuple[Array, Array]:
    return _div(chebmul, input, other)


def chebline(input: Array, other: Array):
    return array([input, other])


def chebmul(
    input: Array,
    other: Array,
    mode: Literal["full", "same", "valid"] = "full",
) -> Array:
    input, other = _as_series(input, other)

    z1 = _c_series_to_z_series(input)
    z2 = _c_series_to_z_series(other)

    output = _z_series_mul(z1, z2, mode=mode)

    output = _z_series_to_c_series(output)

    if mode == "same":
        output = output[: max(input.shape[0], other.shape[0])]

    return output


def chebmulx(
    input: Array,
    mode: Literal["full", "same", "valid"] = "full",
) -> Array:
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
    input: Array,
    exponent: float | Array,
    maximum_exponent: float | Array = 16.0,
) -> Array:
    input = _as_series(input)

    _exponent = int(exponent)

    if _exponent != exponent or _exponent < 0:
        raise ValueError

    if maximum_exponent is not None and _exponent > maximum_exponent:
        raise ValueError

    if _exponent == 0:
        return array([1], dtype=input.dtype)

    if _exponent == 1:
        return input

    output = zeros(input.shape[0] * exponent, dtype=input.dtype)

    output = chebadd(output, input)

    zs = _c_series_to_z_series(input)

    output = _c_series_to_z_series(output)

    for _ in range(2, _exponent + 1):
        output = convolve(output, zs, mode="same")

    output = _z_series_to_c_series(output)

    return output


def chebsub(input: Array, other: Array) -> Array:
    return _subtract(input, other)


def chebval(
    input: Array,
    coefficients: Array,
    tensor: bool = True,
) -> Array:
    coefficients = _as_series(coefficients)

    if tensor:
        coefficients = reshape(
            coefficients,
            coefficients.shape + (1,) * ndim(input),
        )

    match coefficients.shape[0]:
        case 1:
            a = coefficients[0]
            b = 0
        case 2:
            a = coefficients[0]
            b = coefficients[1]
        case _:
            a = coefficients[-2] * ones_like(input)
            b = coefficients[-1] * ones_like(input)

            for i in range(3, coefficients.shape[0] + 1):
                previous = a

                a = coefficients[-i] - b
                b = previous + b * 2.0 * input

    return a + b * input


def hermadd(input: Array, other: Array) -> Array:
    return _add(input, other)


def hermdiv(input: Array, other: Array) -> Tuple[Array, Array]:
    return _div(hermmul, input, other)


def hermeadd(input: Array, other: Array) -> Array:
    return _add(input, other)


def hermediv(input: Array, other: Array) -> Tuple[Array, Array]:
    return _div(hermemul, input, other)


def hermeline(input: float, other: float) -> Array:
    return array([input, other])


def hermemul(
    input: Array,
    other: Array,
    mode: Literal["full", "same", "valid"] = "full",
) -> Array:
    input, other = _as_series(input, other)

    m, n = input.shape[0], other.shape[0]

    if m > n:
        x, y = other, input
    else:
        x, y = input, other

    match x.shape[0]:
        case 1:
            a = hermeadd(zeros(m + n - 1), x[0] * y)
            b = zeros(m + n - 1)
        case 2:
            a = hermeadd(zeros(m + n - 1), x[0] * y)
            b = hermeadd(zeros(m + n - 1), x[1] * y)
        case _:
            size = x.shape[0]

            a = hermeadd(zeros(m + n - 1), x[-2] * y)
            b = hermeadd(zeros(m + n - 1), x[-1] * y)

            for i in range(3, x.shape[0] + 1):
                previous = a

                size = size - 1

                a = hermesub(x[-i] * y, b * (size - 1.0))
                b = hermeadd(previous, hermemulx(b, "same"))

    output = hermeadd(a, hermemulx(b, "same"))

    if mode == "same":
        output = output[: max(m, n)]

    return output


def hermemulx(
    input: Array,
    mode: Literal["full", "same", "valid"] = "full",
) -> Array:
    input = _as_series(input)

    output = zeros(input.shape[0] + 1, dtype=input.dtype)

    output = output.at[1].set(input[0])

    i = arange(1, input.shape[0])

    output = output.at[i + 1].set(input[i])
    output = output.at[i - 1].add(input[i] * i)

    if mode == "same":
        output = output[: input.shape[0]]

    return output


def hermepow(
    input: Array,
    exponent: float | Array,
    maximum_exponent: float | Array = 16.0,
) -> Array:
    return _pow(
        hermemul,
        input,
        exponent,
        maximum_exponent,
    )


def hermesub(input: Array, other: Array) -> Array:
    return _subtract(input, other)


def hermeval(
    input: Array,
    coefficients: Array,
    tensor: bool = True,
):
    coefficients = _as_series(coefficients)

    if tensor:
        coefficients = reshape(
            coefficients,
            coefficients.shape + (1,) * ndim(input),
        )

    if coefficients.shape[0] == 1:
        a = coefficients[0]
        b = 0
    elif coefficients.shape[0] == 2:
        a = coefficients[0]
        b = coefficients[1]
    else:
        size = coefficients.shape[0]

        a = coefficients[-2] * ones_like(input)
        b = coefficients[-1] * ones_like(input)

        for i in range(3, coefficients.shape[0] + 1):
            previous = a

            size = size - 1

            a = coefficients[-i] - b * (size - 1.0)
            b = previous + b * input

    return a + b * input


def hermline(input: Array, other: Array) -> Array:
    return array([input, other / 2])


def hermmul(
    input: Array,
    other: Array,
    mode: Literal["full", "same", "valid"] = "full",
) -> Array:
    input, other = _as_series(input, other)

    m, n = input.shape[0], other.shape[0]

    if m > n:
        x, y = other, input
    else:
        x, y = input, other

    if x.shape[0] == 1:
        a = hermadd(zeros(m + n - 1), x[0] * y)
        b = zeros(m + n - 1)
    elif x.shape[0] == 2:
        a = hermadd(zeros(m + n - 1), x[0] * y)
        b = hermadd(zeros(m + n - 1), x[1] * y)
    else:
        size = x.shape[0]

        a = hermadd(zeros(m + n - 1), x[-2] * y)
        b = hermadd(zeros(m + n - 1), x[-1] * y)

        for i in range(3, x.shape[0] + 1):
            previous = a

            size = size - 1

            a = hermsub(x[-i] * y, b * (2 * (size - 1.0)))
            b = hermadd(previous, hermmulx(b, "same") * 2.0)

    output = hermadd(a, hermmulx(b, "same") * 2)

    if mode == "same":
        output = output[: max(m, n)]

    return output


def hermmulx(
    input: Array,
    mode: Literal["full", "same", "valid"] = "full",
) -> Array:
    input = _as_series(input)
    output = zeros(input.shape[0] + 1, dtype=input.dtype)
    output = output.at[1].set(input[0] / 2.0)

    i = arange(1, input.shape[0])

    output = output.at[i + 1].set(input[i] / 2.0)
    output = output.at[i - 1].add(input[i] * i)

    if mode == "same":
        output = output[: input.shape[0]]

    return output


def hermpow(
    input: Array,
    exponent: float | Array,
    maximum_exponent: float | Array = 16.0,
) -> Array:
    return _pow(
        hermmul,
        input,
        exponent,
        maximum_exponent,
    )


def hermsub(input: Array, other: Array):
    return _subtract(input, other)


def hermval(
    input: Array,
    coefficients: Array,
    tensor: bool = True,
):
    coefficients = _as_series(coefficients)

    if tensor:
        coefficients = reshape(
            coefficients,
            coefficients.shape + (1,) * ndim(input),
        )

    if coefficients.shape[0] == 1:
        a = coefficients[0]
        b = 0
    elif coefficients.shape[0] == 2:
        a = coefficients[0]
        b = coefficients[1]
    else:
        size = coefficients.shape[0]

        a = coefficients[-2] * ones_like(input)
        b = coefficients[-1] * ones_like(input)

        for i in range(3, coefficients.shape[0] + 1):
            previous = a

            size = size - 1

            a = coefficients[-i] - b * (2.0 * (size - 1.0))
            b = previous + b * input * 2.0

    return a + b * input * 2.0


def lagadd(input: Array, other: Array) -> Array:
    return _add(input, other)


def lagdiv(input: Array, other: Array) -> Tuple[Array, Array]:
    return _div(lagmul, input, other)


def lagline(input: Array, other: Array) -> Array:
    return array([input + other, -other])


def lagmul(
    input: Array,
    other: Array,
    mode: Literal["full", "same", "valid"] = "full",
) -> Array:
    input, other = _as_series(input, other)

    m, n = input.shape[0], other.shape[0]

    if m > n:
        x, y = other, input
    else:
        x, y = input, other

    match x.shape[0]:
        case 1:
            a = lagadd(zeros(m + n - 1), x[0] * y)
            b = zeros(m + n - 1)
        case 2:
            a = lagadd(zeros(m + n - 1), x[0] * y)
            b = lagadd(zeros(m + n - 1), x[1] * y)
        case _:
            size = x.shape[0]

            a = lagadd(zeros(m + n - 1), x[-2] * y)
            b = lagadd(zeros(m + n - 1), x[-1] * y)

            for i in range(3, x.shape[0] + 1):
                previous = a

                size = size - 1

                a = lagsub(x[-i] * y, (b * (size - 1.0)) / size)
                b = lagadd(
                    previous, lagsub((2.0 * size - 1.0) * b, lagmulx(b, "same")) / size
                )

    output = lagadd(a, lagsub(b, lagmulx(b, "same")))

    if mode == "same":
        output = output[: max(m, n)]

    return output


def lagmulx(
    input: Array,
    mode: Literal["full", "same", "valid"] = "full",
) -> Array:
    input = _as_series(input)

    output = zeros(input.shape[0] + 1, dtype=input.dtype)

    output = output.at[0].set(+input[0])
    output = output.at[1].set(-input[0])

    i = arange(1, input.shape[0])

    output = output.at[i + 1].set(-input[i] * (i + 1))

    output = output.at[i].add(input[i] * (2 * i + 1))

    output = output.at[i - 1].add(-input[i] * i)

    if mode == "same":
        output = output[: input.shape[0]]

    return output


def lagpow(
    input: Array,
    exponent: float | Array,
    maximum_exponent: float | Array = 16.0,
) -> Array:
    return _pow(
        lagmul,
        input,
        exponent,
        maximum_exponent,
    )


def lagsub(input: Array, other: Array) -> Array:
    return _subtract(input, other)


def lagval(
    input: Array,
    coefficients: Array,
    tensor: bool = True,
):
    coefficients = _as_series(coefficients)

    if tensor:
        coefficients = reshape(
            coefficients,
            coefficients.shape + (1,) * ndim(input),
        )

    match coefficients.shape[0]:
        case 1:
            a = coefficients[0]
            b = 0
        case 2:
            a = coefficients[0]
            b = coefficients[1]
        case _:
            size = coefficients.shape[0]

            a = coefficients[-2] * ones_like(input)
            b = coefficients[-1] * ones_like(input)

            for i in range(3, coefficients.shape[0] + 1):
                previous = a

                size = size - 1

                a = coefficients[-i] - (b * (size - 1.0)) / size
                b = previous + (b * ((2.0 * size - 1.0) - input)) / size

    return a + b * (1 - input)


def legadd(input: Array, other: Array) -> Array:
    return _add(input, other)


def legdiv(input: Array, other: Array) -> Tuple[Array, Array]:
    return _div(legmul, input, other)


def legline(input: float, other: float) -> Array:
    return array([input, other])


def legmul(
    input: Array,
    other: Array,
    mode: Literal["full", "same", "valid"] = "full",
) -> Array:
    input, other = _as_series(input, other)

    m, n = input.shape[0], other.shape[0]

    if m > n:
        x, y = other, input
    else:
        x, y = input, other

    match x.shape[0]:
        case 1:
            a = legadd(zeros(m + n - 1), x[0] * y)
            b = zeros(m + n - 1)
        case 2:
            a = legadd(zeros(m + n - 1), x[0] * y)
            b = legadd(zeros(m + n - 1), x[1] * y)
        case _:
            size = x.shape[0]

            a = legadd(zeros(m + n - 1), x[-2] * y)
            b = legadd(zeros(m + n - 1), x[-1] * y)

            for i in range(3, x.shape[0] + 1):
                previous = a

                size = size - 1

                a = legsub(x[-i] * y, (b * (size - 1.0)) / size)
                b = legadd(previous, (legmulx(b, "same") * (2.0 * size - 1.0)) / size)

    output = legadd(a, legmulx(b, "same"))

    if mode == "same":
        output = output[: max(m, n)]

    return output


def legmulx(
    input: Array,
    mode: Literal["full", "same"] = "full",
) -> Array:
    input = _as_series(input)

    output = zeros(input.shape[0] + 1, dtype=input.dtype).at[1].set(input[0])

    for i in range(1, input.shape[0]):
        output = output.at[i + 1].set((input[i] * (i + 1)) / (i + i + 1))
        output = output.at[i - 1].add((input[i] * (i + 0)) / (i + i + 1))

    if mode == "same":
        output = output[: input.shape[0]]

    return output


def legpow(
    input: Array,
    exponent: float | Array,
    maximum_exponent: float | Array = 16.0,
) -> Array:
    return _pow(
        legmul,
        input,
        exponent,
        maximum_exponent,
    )


def legsub(input: Array, other: Array) -> Array:
    return _subtract(input, other)


def legval(
    input: Array,
    coefficients: Array,
    tensor: bool = True,
) -> Array:
    coefficients = _as_series(coefficients)

    if tensor:
        coefficients = reshape(
            coefficients,
            coefficients.shape + (1,) * ndim(input),
        )

    match coefficients.shape[0]:
        case 1:
            a = coefficients[0]
            b = 0
        case 2:
            a = coefficients[0]
            b = coefficients[1]
        case _:
            size = coefficients.shape[0]

            a = coefficients[-2] * ones_like(input)
            b = coefficients[-1] * ones_like(input)

            for i in range(3, coefficients.shape[0] + 1):
                previous = a

                size = size - 1

                a = coefficients[-i] - (b * (size - 1.0)) / size

                b = previous + (b * input * (2.0 * size - 1.0)) / size

    return a + b * input


def polyadd(input: Array, other: Array) -> Array:
    return _add(input, other)


def polydiv(input: Array, other: Array) -> Tuple[Array, Array]:
    input, other = _as_series(input, other)

    return _div(polymul, input, other)


def polyline(input: float, other: float) -> Array:
    return array([input, other])


def polymul(
    input: Array,
    other: Array,
    mode: Literal["full", "same", "valid"] = "full",
) -> Array:
    input, other = _as_series(input, other)

    output = convolve(input, other)

    if mode == "same":
        output = output[: max(input.shape[0], other.shape[0])]

    return output


def polymulx(
    input: Array,
    mode: Literal["full", "same", "valid"] = "full",
) -> Array:
    input = _as_series(input)

    output = zeros(input.shape[0] + 1, dtype=input.dtype)

    output = output.at[1:].set(input)

    if mode == "same":
        output = output[: input.shape[0]]

    return output


def polypow(
    input: Array,
    exponent: float | Array,
    maximum_exponent: float | Array = 16.0,
) -> Array:
    return _pow(
        polymul,
        input,
        exponent,
        maximum_exponent,
    )


def polysub(input: Array, other: Array) -> Array:
    return _subtract(input, other)


def polyval(
    input: Array,
    coefficients: Array,
    tensor: bool = True,
) -> Array:
    coefficients = _as_series(coefficients)

    if tensor:
        coefficients = reshape(
            coefficients,
            coefficients.shape + (1,) * ndim(input),
        )

    output = coefficients[-1] + zeros_like(input)

    for i in range(2, coefficients.shape[0] + 1):
        output = coefficients[-i] + output * input

    return output


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
