import functools
import math
import operator
from typing import Callable, List, Literal, Tuple

import torch
import torch._numpy._funcs_impl
import torch.linalg
import torchaudio.functional
from torch import (
    Tensor,
    abs,
    any,
    arange,
    atleast_1d,
    concatenate,
    finfo,
    flip,
    full,
    imag,
    moveaxis,
    nonzero,
    ones,
    ones_like,
    promote_types,
    real,
    reshape,
    roll,
    sort,
    sqrt,
    stack,
    sum,
    tensor,
    where,
    zeros,
    zeros_like,
)

torch.set_default_dtype(torch.float64)

chebdomain = torch.tensor([-1.0, 1.0])
chebone = torch.tensor([1.0])
chebx = torch.tensor([0.0, 1.0])
chebzero = torch.tensor([0.0])
hermdomain = torch.tensor([-1.0, 1.0])
hermedomain = torch.tensor([-1.0, 1.0])
hermeone = torch.tensor([1.0])
hermex = torch.tensor([0.0, 1.0])
hermezero = torch.tensor([0.0])
hermone = torch.tensor([1.0])
hermx = torch.tensor([0.0, 1.0 / 2.0])
hermzero = torch.tensor([0.0])
lagdomain = torch.tensor([0.0, 1.0])
lagone = torch.tensor([1.0])
lagx = torch.tensor([1.0, -1.0])
lagzero = torch.tensor([0.0])
legdomain = torch.tensor([-1.0, 1.0])
legone = torch.tensor([1.0])
legx = torch.tensor([0.0, 1.0])
legzero = torch.tensor([0.0])
polydomain = torch.tensor([-1.0, 1.0])
polyone = torch.tensor([1.0])
polyx = torch.tensor([0.0, 1.0])
polyzero = torch.tensor([0.0])


def _nonzero(input, size=1, fill_value=0):
    output = torch.nonzero(input, as_tuple=False)

    if output.shape[0] > size:
        output = output[:size]
    elif output.shape[0] < size:
        output = concatenate(
            [
                output,
                full(
                    [
                        size - output.shape[0],
                        output.shape[1],
                    ],
                    fill_value,
                ),
            ],
            0,
        )

    return output


def _add(input: Tensor, other: Tensor) -> Tensor:
    [input, other] = _as_series([input, other])

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


def _as_series(items: List[Tensor], trim: bool = False) -> List[Tensor]:
    outputs = []

    for item in items:
        output = atleast_1d(item)

        if trim:
            output = _trim_sequence(output)

        outputs = [
            *outputs,
            output,
        ]

    dtype = outputs[0].dtype

    for output in outputs[1:]:
        dtype = promote_types(dtype, output.dtype)

    for index, output in enumerate(outputs):
        if output.dtype != dtype:
            outputs[index] = output.astype(dtype)

    return outputs


def _c_series_to_z_series(input: Tensor) -> Tensor:
    n = math.prod(input.shape)

    zs = zeros(2 * n - 1, dtype=input.dtype)

    zs[n - 1 :] = input / 2.0

    return flip(zs, [0]) + zs


def _div(func: Callable, input: Tensor, other: Tensor) -> Tuple[Tensor, Tensor]:
    [input, other] = _as_series([input, other])

    m = input.shape[0]
    n = other.shape[0]

    if m < n:
        return zeros_like(input[:1]), input

    if n == 1:
        return input / other[-1], zeros_like(input[:1])

    def f(x: Tensor) -> Tensor:
        indicies = flip(x, [0])

        indicies = _nonzero(indicies, size=1)

        return x.shape[0] - 1 - indicies[0][0]

    quotient = zeros(m - n + 1, dtype=input.dtype)

    ridx = input.shape[0] - 1

    size = m - f(other) - 1

    y = zeros(m + n + 1, dtype=input.dtype)

    y[size] = 1.0

    x = quotient, input, y, ridx

    for index in range(0, size):
        quotient, remainder, y2, ridx1 = x

        j = size - index

        p = func(y2, other)

        pidx = f(p)

        t = remainder[ridx1] / p[pidx]

        a = remainder.at[ridx1].set(0.0)
        b = t * p.at[pidx].set(0.0)

        remainder = _subtract(a, b)
        remainder = remainder[: remainder.shape[0]]

        quotient[j] = t

        ridx1 = ridx1 - 1

        y2 = roll(y2, -1)

        x = quotient, remainder, y2, ridx1

    quotient, remainder, _, _ = x

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

    vandermonde = vandermonde_func(input, degree[-1])[:, degree].T

    if weight is not None:
        if weight.ndim != 1:
            raise TypeError

        if input.shape[0] != weight.shape[0]:
            raise TypeError

        vandermonde = vandermonde * weight

        b = other.T * weight
    else:
        b = other.T

    if relative_condition is None:
        relative_condition = input.shape[0] * finfo(input.dtype).eps

    if torch.is_complex(vandermonde):
        scale = vandermonde.real**2.0 + vandermonde.imag**2.0
        scale = sum(scale, dim=1)
        scale = sqrt(scale)
    else:
        scale = vandermonde**2.0
        scale = sum(scale, dim=1)
        scale = sqrt(scale)

    scale = where(scale == 0, 1, scale)

    output, residuals, rank, scale = torch.linalg.lstsq(
        vandermonde.T / scale,
        b.T,
        relative_condition,
    )

    output = (output.T / scale).T

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


def _from_roots(f: Callable, g: Callable, input: Tensor) -> Tensor:
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

        previous = tensor([zeros(input.shape[0] + 1, dtype=p.dtype)] * len(p))

        y = previous

        for i in range(0, m):
            y[i] = g(z[i], z[i + m])[: input.shape[0] + 1]

        previous = y

        if r:
            previous[0] = g(previous[0], z[2 * m])[: input.shape[0] + 1]

        x = m, previous

    _, output = x

    return output[0]


def _get_domain(x: Tensor) -> Tensor:
    if any(torch.is_complex(x)):
        rmin, rmax = real(x).min(), real(x).max()
        imin, imax = imag(x).min(), imag(x).max()

        return tensor(((rmin + 1.0j * imin), (rmax + 1.0j * imax)))

    return tensor((x.min(), x.max()))


def _map_domain(x: Tensor, y: Tensor, z: Tensor) -> Tensor:
    (a, b), (c, d) = y, z

    return (b * c - a * d) / (b - a) + (d - c) / (b - a) * x


def _map_parameters(input: Tensor, other: Tensor) -> Tuple[Tensor, Tensor]:
    a = input[1] - input[0]
    b = other[1] - other[0]

    x = (input[1] * other[0] - input[0] * other[1]) / a
    y = b / a

    return x, y


def _normed_hermite_e_n(x: Tensor, n) -> Tensor:
    if n == 0:
        output = full(x.shape, 1.0 / math.sqrt(math.sqrt(2.0 * math.pi)))
    else:
        a = zeros_like(x)
        b = ones_like(x) / math.sqrt(math.sqrt(2.0 * math.pi))

        size = tensor(n)

        for _ in range(0, n - 1):
            previous = a

            a = -b * sqrt((size - 1.0) / size)

            b = previous + b * x * sqrt(1.0 / size)

            size = size - 1.0

        output = a + b * x

    return output


def _normed_hermite_n(x: Tensor, n) -> Tensor:
    if n == 0:
        output = full(x.shape, 1 / math.sqrt(math.sqrt(math.pi)))
    else:
        a = zeros_like(x)

        b = ones_like(x) / math.sqrt(math.sqrt(math.pi))

        size = tensor(n)

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


def _pad_along_axis(input: Tensor, padding=(0, 0), axis=0):
    input = moveaxis(input, axis, 0)

    if padding[0] < 0:
        input = input[abs(padding[0]) :]
        padding = (0, padding[1])
    if padding[1] < 0:
        input = input[: -abs(padding[1])]
        padding = (padding[0], 0)

    npad = torch.tensor([(0, 0)] * input.ndim)

    npad[0] = padding

    output = torch._numpy._funcs_impl.pad(
        input, pad_width=npad, mode="constant", constant_values=0
    )

    return moveaxis(output, 0, axis)


def _pow(
    func: Callable,
    input: Tensor,
    exponent: int | Tensor,
    maximum_exponent: int | Tensor,
) -> Tensor:
    [input] = _as_series([input])

    _exponent = int(exponent)

    if _exponent != exponent or _exponent < 0:
        raise ValueError

    if maximum_exponent is not None and _exponent > maximum_exponent:
        raise ValueError

    match _exponent:
        case 0:
            output = tensor([1], dtype=input.dtype)
        case 1:
            output = input
        case _:
            output = zeros(input.shape[0] * exponent, dtype=input.dtype)

            output = _add(output, input)

            for _ in range(2, _exponent + 1):
                output = func(output, input, mode="same")

    return output


def _subtract(input: Tensor, other: Tensor) -> Tensor:
    [input, other] = _as_series([input, other])

    if input.shape[0] > other.shape[0]:
        output = -other

        output = concatenate(
            [
                output,
                zeros(
                    input.shape[0] - other.shape[0],
                    dtype=other.dtype,
                ),
            ],
        )
        output = input + output
    else:
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

    [input] = _as_series([input])

    indices = nonzero(abs(input) > tol)

    if indices.shape[0] == 0:
        output = input[:1] * 0
    else:
        output = input[: indices[-1] + 1]

    return output


def _trim_sequence(input: Tensor) -> Tensor:
    if input.shape[0] == 0:
        output = input
    else:
        index = 0

        for index in range(input.shape[0] - 1, -1, -1):
            if input[index] != 0:
                break

        output = input[: index + 1]

    return output


def _vandermonde(vander_fs, points: Tensor, degrees: Tensor) -> Tensor:
    n_dims = len(vander_fs)

    if n_dims != len(points):
        raise ValueError

    if n_dims != len(degrees):
        raise ValueError

    if n_dims == 0:
        raise ValueError

    points = tuple(tensor(tuple(points)) + 0.0)

    output = []

    for i in range(n_dims):
        vandermonde = vander_fs[i](points[i], degrees[i])

        vandermonde = vandermonde[(..., *_nth_slice(i, n_dims))]

        output = [*output, vandermonde]

    return functools.reduce(operator.mul, output)


def _z_series_mul(
    input: Tensor,
    other: Tensor,
    mode: Literal["full", "same", "valid"] = "full",
) -> Tensor:
    return torchaudio.functional.convolve(input, other, mode=mode)


def _z_series_to_c_series(input: Tensor) -> Tensor:
    n = (math.prod(input.shape) + 1) // 2

    c = input[n - 1 :]

    c[1:n] = c[1:n] * 2.0

    return c


def chebadd(input: Tensor, other: Tensor) -> Tensor:
    return _add(input, other)


def chebdiv(input: Tensor, other: Tensor) -> Tuple[Tensor, Tensor]:
    return _div(chebmul, input, other)


def chebline(input: float, other: float) -> Tensor:
    return torch.tensor([input, other])


def chebmul(
    input: Tensor,
    other: Tensor,
    mode: Literal["full", "same", "valid"] = "full",
) -> Tensor:
    [input, other] = _as_series([input, other])

    a = _c_series_to_z_series(input)
    b = _c_series_to_z_series(other)

    output = _z_series_mul(a, b, mode=mode)

    output = _z_series_to_c_series(output)

    if mode == "same":
        output = output[: max(input.shape[0], other.shape[0])]

    return output


def chebmulx(
    input: Tensor,
    mode: Literal["full", "same", "valid"] = "full",
) -> Tensor:
    [input] = _as_series([input])

    output = zeros(input.shape[0] + 1, dtype=input.dtype)

    output[1] = input[0]

    if input.shape[0] > 1:
        output[2:] = input[1:] / 2

        output[0:-2] = output[0:-2] + input[1:] / 2

    if mode == "same":
        output = output[: input.shape[0]]

    return output


def chebpow(
    input: Tensor,
    exponent: float | Tensor,
    maximum_exponent: float | Tensor = 16.0,
) -> Tensor:
    [input] = _as_series([input])

    _exponent = int(exponent)

    if _exponent != exponent or _exponent < 0:
        raise ValueError

    if maximum_exponent is not None and _exponent > maximum_exponent:
        raise ValueError

    if _exponent == 0:
        return tensor([1.0], dtype=input.dtype)

    if _exponent == 1:
        return input

    output = zeros(input.shape[0] * exponent, dtype=input.dtype)

    output = chebadd(output, input)

    zs = _c_series_to_z_series(input)

    output = _c_series_to_z_series(output)

    for _ in range(2, _exponent + 1):
        output = torchaudio.functional.convolve(output, zs, mode="same")

    output = _z_series_to_c_series(output)

    return output


def chebsub(input: Tensor, other: Tensor) -> Tensor:
    return _subtract(input, other)


def chebval(
    input: Tensor,
    coefficients: Tensor,
    tensor: bool = True,
) -> Tensor:
    [coefficients] = _as_series([coefficients])

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


def hermdiv(input: Tensor, other: Tensor) -> Tuple[Tensor, Tensor]:
    return _div(hermmul, input, other)


def hermeadd(input: Tensor, other: Tensor) -> Tensor:
    return _add(input, other)


def hermediv(input: Tensor, other: Tensor) -> Tuple[Tensor, Tensor]:
    return _div(hermemul, input, other)


def hermeline(input: float, other: float) -> Tensor:
    return torch.tensor([input, other])


def hermemul(
    input: Tensor,
    other: Tensor,
    mode: Literal["full", "same", "valid"] = "full",
) -> Tensor:
    [input, other] = _as_series([input, other])

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
    input: Tensor,
    mode: Literal["full", "same", "valid"] = "full",
) -> Tensor:
    [input] = _as_series([input])

    output = zeros(input.shape[0] + 1, dtype=input.dtype)

    output[1] = input[0]

    index = arange(1, input.shape[0])

    output[index + 1] = input[index]
    output[index - 1] = output[index - 1] + input[index] * index

    if mode == "same":
        output = output[: input.shape[0]]

    return output


def hermepow(
    input: Tensor,
    exponent: float | Tensor,
    maximum_exponent: float | Tensor = 16.0,
) -> Tensor:
    return _pow(
        hermemul,
        input,
        exponent,
        maximum_exponent,
    )


def hermesub(input: Tensor, other: Tensor) -> Tensor:
    return _subtract(input, other)


def hermeval(
    input: Tensor,
    coefficients: Tensor,
    tensor: bool = True,
):
    [coefficients] = _as_series([coefficients])

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
    return torch.tensor([input, other / 2])


def hermmul(
    input: Tensor,
    other: Tensor,
    mode: Literal["full", "same", "valid"] = "full",
) -> Tensor:
    [input, other] = _as_series([input, other])

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
    input: Tensor,
    mode: Literal["full", "same", "valid"] = "full",
) -> Tensor:
    [input] = _as_series([input])

    output = zeros(input.shape[0] + 1, dtype=input.dtype)

    output[1] = input[0] / 2.0

    i = arange(1, input.shape[0])

    output[i + 1] = input[i] / 2.0
    output[i - 1] = output[i - 1] + input[i] * i

    if mode == "same":
        output = output[: input.shape[0]]

    return output


def hermpow(
    input: Tensor,
    exponent: float | Tensor,
    maximum_exponent: float | Tensor = 16.0,
) -> Tensor:
    return _pow(
        hermmul,
        input,
        exponent,
        maximum_exponent,
    )


def hermsub(input: Tensor, other: Tensor):
    return _subtract(input, other)


def hermval(
    input: Tensor,
    coefficients: Tensor,
    tensor: bool = True,
):
    [coefficients] = _as_series([coefficients])

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


def lagdiv(input: Tensor, other: Tensor) -> Tuple[Tensor, Tensor]:
    return _div(lagmul, input, other)


def lagline(input: float, other: float) -> Tensor:
    return torch.tensor([input + other, -other])


def lagmul(
    input: Tensor,
    other: Tensor,
    mode: Literal["full", "same", "valid"] = "full",
) -> Tensor:
    [input, other] = _as_series([input, other])

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
    input: Tensor,
    mode: Literal["full", "same", "valid"] = "full",
) -> Tensor:
    [input] = _as_series([input])

    output = zeros(input.shape[0] + 1, dtype=input.dtype)

    output[0] = +input[0]
    output[1] = -input[0]

    i = arange(1, input.shape[0])

    output[i + 1] = -input[i] * (i + 1)

    output[i] = output[i] + input[i] * (2 * i + 1)

    output[i - 1] = output[i - 1] - input[i] * i

    if mode == "same":
        output = output[: input.shape[0]]

    return output


def lagpow(
    input: Tensor,
    exponent: float | Tensor,
    maximum_exponent: float | Tensor = 16.0,
) -> Tensor:
    return _pow(
        lagmul,
        input,
        exponent,
        maximum_exponent,
    )


def lagsub(input: Tensor, other: Tensor) -> Tensor:
    return _subtract(input, other)


def lagval(
    input: Tensor,
    coefficients: Tensor,
    tensor: bool = True,
):
    [coefficients] = _as_series([coefficients])

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


def legdiv(input: Tensor, other: Tensor) -> Tuple[Tensor, Tensor]:
    return _div(legmul, input, other)


def legline(input: float, other: float) -> Tensor:
    return torch.tensor([input, other])


def legmul(
    input: Tensor,
    other: Tensor,
    mode: Literal["full", "same", "valid"] = "full",
) -> Tensor:
    [input, other] = _as_series([input, other])

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
    input: Tensor,
    mode: Literal["full", "same"] = "full",
) -> Tensor:
    [input] = _as_series([input])

    output = zeros(input.shape[0] + 1, dtype=input.dtype)
    output[1] = input[0]

    for i in range(1, input.shape[0]):
        output[i + 1] = (input[i] * (i + 1)) / (i + i + 1)
        output[i - 1] = output[i - 1] + (input[i] * (i + 0)) / (i + i + 1)

    if mode == "same":
        output = output[: input.shape[0]]

    return output


def legpow(
    input: Tensor,
    exponent: float | Tensor,
    maximum_exponent: float | Tensor = 16.0,
) -> Tensor:
    return _pow(
        legmul,
        input,
        exponent,
        maximum_exponent,
    )


def legsub(input: Tensor, other: Tensor) -> Tensor:
    return _subtract(input, other)


def legval(
    input: Tensor,
    coefficients: Tensor,
    tensor: bool = True,
) -> Tensor:
    [coefficients] = _as_series([coefficients])

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


def polydiv(input: Tensor, other: Tensor) -> Tuple[Tensor, Tensor]:
    [input, other] = _as_series([input, other])

    return _div(polymul, input, other)


def polyline(input: float, other: float) -> Tensor:
    return torch.tensor([input, other])


def polymul(
    input: Tensor,
    other: Tensor,
    mode: Literal["full", "same", "valid"] = "full",
) -> Tensor:
    [input, other] = _as_series([input, other])

    output = torchaudio.functional.convolve(input, other)

    if mode == "same":
        output = output[: max(input.shape[0], other.shape[0])]

    return output


def polymulx(
    input: Tensor,
    mode: Literal["full", "same", "valid"] = "full",
) -> Tensor:
    [input] = _as_series([input])

    output = zeros(input.shape[0] + 1, dtype=input.dtype)

    output[1:] = input

    if mode == "same":
        output = output[: input.shape[0]]

    return output


def polypow(
    input: Tensor,
    exponent: float | Tensor,
    maximum_exponent: float | Tensor = 16.0,
) -> Tensor:
    return _pow(
        polymul,
        input,
        exponent,
        maximum_exponent,
    )


def polysub(input: Tensor, other: Tensor) -> Tensor:
    return _subtract(input, other)


def polyval(
    input: Tensor,
    coefficients: Tensor,
    tensor: bool = True,
) -> Tensor:
    [coefficients] = _as_series([coefficients])

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
