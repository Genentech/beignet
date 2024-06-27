import functools
import math
import operator
from typing import Callable, Literal, Tuple

from jax import Array
from jax.numpy import (
    abs,
    any,
    arange,
    array,
    concatenate,
    convolve,
    finfo,
    flip,
    full,
    imag,
    iscomplex,
    moveaxis,
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
from torch import (
    Tensor,
    tensor,
)

chebdomain = tensor([-1.0, 1.0])
chebone = tensor([1.0])
chebx = tensor([0.0, 1.0])
chebzero = tensor([0.0])
hermdomain = tensor([-1.0, 1.0])
hermedomain = tensor([-1.0, 1.0])
hermeone = tensor([1.0])
hermex = tensor([0.0, 1.0])
hermezero = tensor([0.0])
hermone = tensor([1.0])
hermx = tensor([0.0, 1.0 / 2.0])
hermzero = tensor([0.0])
lagdomain = tensor([0.0, 1.0])
lagone = tensor([1.0])
lagx = tensor([1.0, -1.0])
lagzero = tensor([0.0])
legdomain = tensor([-1.0, 1.0])
legone = tensor([1.0])
legx = tensor([0.0, 1.0])
legzero = tensor([0.0])
polydomain = tensor([-1.0, 1.0])
polyone = tensor([1.0])
polyx = tensor([0.0, 1.0])
polyzero = tensor([0.0])


def _add(input: Tensor, other: Tensor) -> Tensor:
    input, other = _as_series(input, other)

    if input.shape[0] > other.shape[0]:
        output = concatenate(
            [
                other,
                zeros(
                    input.shape[0] - other.shape[0],
                    dtype=other.dtype,
                ),
            ],
        )

        output = input + output
    else:
        output = concatenate(
            [
                input,
                zeros(
                    other.shape[0] - input.shape[0],
                    dtype=input.dtype,
                ),
            ]
        )

        output = other + output

    return output


def _as_series(*args, trim: bool = False) -> Tuple[Array, ...]:
    xs = ()

    for arg in args:
        x = array(arg)

        if x.ndim == 0:
            x = ravel(x)

        if trim:
            x = _trim_sequence(x)

        xs = *xs, x

    # xs = promote_dtypes_inexact(*xs)

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

    def f(x: Array) -> Array:
        indicies = flip(x, axis=0)

        indicies = nonzero(indicies, size=1)

        return x.shape[0] - 1 - indicies[0][0]

    q = zeros(m - n + 1, dtype=input.dtype)

    ridx = input.shape[0] - 1

    sz = m - f(other) - 1

    y = zeros(m + n + 1, dtype=input.dtype)
    y = y.at[sz].set(1.0)

    x = q, input, y, ridx

    for index in range(0, sz):
        q, r, y2, ridx1 = x

        j = sz - index

        p = func(y2, other)

        pidx = f(p)

        t = r[ridx1] / p[pidx]

        a = r.at[ridx1].set(0.0)
        b = t * p.at[pidx].set(0.0)

        r = _subtract(a, b)
        r = r[: r.shape[0]]

        q = q.at[j].set(t)

        ridx1 = ridx1 - 1

        y2 = roll(y2, -1)

        x = q, r, y2, ridx1

    q, r, _, _ = x

    return q, r


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
    if degree.ndim > 1 or math.prod(degree.shape) == 0:
        raise TypeError

    # if torch.is_floating_point(degree) or degree.is_complex(degree):
    #     raise TypeError

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

    degree = sort(degree)

    vandermonde = transpose(vandermonde_func(input, degree[-1])[:, degree])

    if weight is not None:
        if weight.ndim != 1:
            raise TypeError

        if input.shape[0] != weight.shape[0]:
            raise TypeError

        vandermonde = vandermonde * weight

        b = transpose(other) * weight
    else:
        b = transpose(other)

    if relative_condition is None:
        relative_condition = input.shape[0] * finfo(input.dtype).eps

    if iscomplex(vandermonde):
        scale = sqrt(sum(square(real(vandermonde)) + square(imag(vandermonde)), axis=1))
    else:
        scale = sqrt(sum(square(vandermonde), axis=1))

    scale = where(scale == 0, 1, scale)

    output, residuals, rank, scale = lstsq(
        transpose(vandermonde) / scale,
        transpose(b),
        relative_condition,
    )

    output = transpose(transpose(output) / scale)

    if degree.ndim > 0:
        if output.ndim == 2:
            x = zeros(
                [degree[-1] + 1, output.shape[1]],
                dtype=output.dtype,
            )
        else:
            x = zeros(
                degree[-1] + 1,
                dtype=output.dtype,
            )

        output = x.at[degree].set(output)

    if full:
        return output, [residuals, rank, scale, relative_condition]
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
    if any(iscomplex(x)):
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

    npad = [(0, 0)] * input.ndim

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

        output = input + concatenate(
            [
                output,
                zeros(
                    input.shape[0] - other.shape[0],
                    dtype=other.dtype,
                ),
            ],
        )

        return output

    output = -other

    output = concatenate(
        [
            output[: input.shape[0]] + input,
            output[input.shape[0] :],
        ],
    )

    return output


def _trim_coefficients(input: Tensor, tol: float = 0.0) -> Tensor:
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

    return c.at[1:n].multiply(2.0)


def chebadd(input: Tensor, other: Tensor) -> Tensor:
    return _add(input, other)


def chebdiv(input: Array, other: Array) -> Tuple[Array, Array]:
    return _div(chebmul, input, other)


def chebline(input: float, other: float) -> Tensor:
    return tensor([input, other])


def chebmul(
    input: Array,
    other: Array,
    mode: Literal["full", "same", "valid"] = "full",
) -> Array:
    input, other = _as_series(input, other)

    a = _c_series_to_z_series(input)
    b = _c_series_to_z_series(other)

    output = _z_series_mul(a, b, mode=mode)

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
        return array([1.0], dtype=input.dtype)

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
            coefficients.shape + (1,) * input.ndim,
        )

    match coefficients.shape[0]:
        case 1:
            a = coefficients[0]
            b = 0.0
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


def hermadd(input: Tensor, other: Tensor) -> Tensor:
    return _add(input, other)


def hermdiv(input: Array, other: Array) -> Tuple[Array, Array]:
    return _div(hermmul, input, other)


def hermeadd(input: Tensor, other: Tensor) -> Tensor:
    return _add(input, other)


def hermediv(input: Array, other: Array) -> Tuple[Array, Array]:
    return _div(hermemul, input, other)


def hermeline(input: float, other: float) -> Tensor:
    return tensor([input, other])


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
            coefficients.shape + (1,) * input.ndim,
        )

    match coefficients.shape[0]:
        case 1:
            a = coefficients[0]
            b = 0.0
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

                a = coefficients[-i] - b * (size - 1.0)

                b = previous + b * input

    return a + b * input


def hermline(input: float, other: float) -> Tensor:
    return tensor([input, other / 2])


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

    match x.shape[0]:
        case 1:
            a = hermadd(zeros(m + n - 1), x[0] * y)
            b = zeros(m + n - 1)
        case 2:
            a = hermadd(zeros(m + n - 1), x[0] * y)
            b = hermadd(zeros(m + n - 1), x[1] * y)
        case _:
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
            coefficients.shape + (1,) * input.ndim,
        )

    match coefficients.shape[0]:
        case 1:
            a = coefficients[0]
            b = 0.0
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

                a = coefficients[-i] - b * (2.0 * (size - 1.0))

                b = previous + b * input * 2.0

    return a + b * input * 2.0


def lagadd(input: Tensor, other: Tensor) -> Tensor:
    return _add(input, other)


def lagdiv(input: Array, other: Array) -> Tuple[Array, Array]:
    return _div(lagmul, input, other)


def lagline(input: float, other: float) -> Tensor:
    return tensor([input + other, -other])


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
            coefficients.shape + (1,) * input.ndim,
        )

    match coefficients.shape[0]:
        case 1:
            a = coefficients[0]
            b = 0.0
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

    return a + b * (1.0 - input)


def legadd(input: Tensor, other: Tensor) -> Tensor:
    return _add(input, other)


def legdiv(input: Array, other: Array) -> Tuple[Array, Array]:
    return _div(legmul, input, other)


def legline(input: float, other: float) -> Tensor:
    return tensor([input, other])


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
            coefficients.shape + (1,) * input.ndim,
        )

    match coefficients.shape[0]:
        case 1:
            a = coefficients[0]
            b = 0.0
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


def polyadd(input: Tensor, other: Tensor) -> Tensor:
    return _add(input, other)


def polydiv(input: Array, other: Array) -> Tuple[Array, Array]:
    input, other = _as_series(input, other)

    return _div(polymul, input, other)


def polyline(input: float, other: float) -> Tensor:
    return tensor([input, other])


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
            coefficients.shape + (1,) * input.ndim,
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
    "chebval",
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
    "hermeval",
    "hermex",
    "hermezero",
    "hermline",
    "hermmul",
    "hermmulx",
    "hermone",
    "hermpow",
    "hermsub",
    "hermtrim",
    "hermval",
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
    "lagval",
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
    "legval",
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
    "polyval",
    "polyx",
    "polyzero",
]
