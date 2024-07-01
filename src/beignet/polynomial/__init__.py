# cheb  : chebyshev_series
# herm  : physicists_hermite_series
# herme : probabilists_hermite_series
# lag   : laguerre_series
# leg   : legendre_series
# poly  : power_series

import functools
import math
import operator
from typing import Callable, Literal, Tuple

import numpy
import torch
import torch._numpy._funcs_impl
import torch.linalg
import torchaudio.functional
from torch import Tensor

from .__as_series import _as_series
from ._chebadd import chebadd
from ._chebcompanion import chebcompanion
from ._chebsub import chebsub
from ._hermadd import hermadd
from ._hermcompanion import hermcompanion
from ._hermeadd import hermeadd
from ._hermecompanion import hermecompanion
from ._hermeline import hermeline
from ._hermemul import hermemul
from ._hermemulx import hermemulx
from ._hermesub import hermesub
from ._hermeval import hermeval
from ._hermevander import hermevander
from ._hermline import hermline
from ._hermmul import hermmul
from ._hermmulx import hermmulx
from ._hermsub import hermsub
from ._hermval import hermval
from ._hermvander import hermvander
from ._lagadd import lagadd
from ._lagcompanion import lagcompanion
from ._lagline import lagline
from ._lagmul import lagmul
from ._lagmulx import lagmulx
from ._lagsub import lagsub
from ._lagval import lagval
from ._lagvander import lagvander
from ._legadd import legadd
from ._legcompanion import legcompanion
from ._legline import legline
from ._legmul import legmul
from ._legmulx import legmulx
from ._legsub import legsub
from ._legval import legval
from ._legvander import legvander
from ._polyadd import polyadd
from ._polycompanion import polycompanion
from ._polyline import polyline
from ._polymul import polymul
from ._polymulx import polymulx
from ._polysub import polysub
from ._polyval import polyval
from ._polyvander import polyvander

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


def _c_series_to_z_series(
    input: Tensor,
) -> Tensor:
    index = math.prod(input.shape)

    zs = torch.zeros(2 * index - 1, dtype=input.dtype)

    zs[index - 1 :] = input / 2.0

    return torch.flip(zs, dims=[0]) + zs


def _div(
    func: Callable,
    input: Tensor,
    other: Tensor,
) -> Tuple[Tensor, Tensor]:
    [input, other] = _as_series([input, other])

    m = input.shape[0]
    n = other.shape[0]

    if m < n:
        return torch.zeros_like(input[:1]), input

    if n == 1:
        return input / other[-1], torch.zeros_like(input[:1])

    def f(x: Tensor) -> Tensor:
        indicies = torch.flip(x, [0])

        indicies = _nonzero(indicies, size=1)

        return x.shape[0] - 1 - indicies[0][0]

    quotient = torch.zeros(m - n + 1, dtype=input.dtype)

    ridx = input.shape[0] - 1

    size = m - f(other) - 1

    y = torch.zeros(m + n + 1, dtype=input.dtype)

    y[size] = 1.0

    x = quotient, input, y, ridx

    for index in range(0, size):
        quotient, remainder, y2, ridx1 = x

        j = size - index

        p = func(y2, other)

        pidx = f(p)

        t = remainder[ridx1] / p[pidx]

        remainder_modified = remainder.clone()
        remainder_modified[ridx1] = 0.0

        a = remainder_modified

        p_modified = p.clone()
        p_modified[pidx] = 0.0

        b = t * p_modified

        [a, b] = _as_series([a, b])

        if a.shape[0] > b.shape[0]:
            output = -b

            output = torch.concatenate(
                [
                    output,
                    torch.zeros(
                        a.shape[0] - b.shape[0],
                        dtype=b.dtype,
                    ),
                ],
            )
            output = a + output
        else:
            output = -b

            output = torch.concatenate(
                [
                    output[: a.shape[0]] + a,
                    output[a.shape[0] :],
                ],
            )

        remainder = output

        remainder = remainder[: remainder.shape[0]]

        quotient[j] = t

        ridx1 = ridx1 - 1

        y2 = torch.roll(y2, -1)

        x = quotient, remainder, y2, ridx1

    quotient, remainder, _, _ = x

    return quotient, remainder


def _evaluate(
    func: Callable,
    input: Tensor,
    *args,
) -> Tensor:
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

    degree, _ = torch.sort(degree)

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
        relative_condition = input.shape[0] * torch.finfo(input.dtype).eps

    if torch.is_complex(vandermonde):
        scale = vandermonde.real**2.0 + vandermonde.imag**2.0
        scale = torch.sum(scale, dim=1)
        scale = torch.sqrt(scale)
    else:
        scale = vandermonde**2.0
        scale = torch.sum(scale, dim=1)
        scale = torch.sqrt(scale)

    scale = torch.where(scale == 0, 1, scale)

    output, residuals, rank, scale = torch.linalg.lstsq(
        vandermonde.T / scale,
        b.T,
        relative_condition,
    )

    output = (output.T / scale).T

    if degree.ndim > 0:
        if output.ndim == 2:
            x = torch.zeros(
                [degree[-1] + 1, output.shape[1]],
                dtype=output.dtype,
            )
        else:
            x = torch.zeros(
                degree[-1] + 1,
                dtype=output.dtype,
            )

        output = x.at[degree].set(output)

    if full:
        return output, [residuals, rank, scale, relative_condition]
    else:
        return output


def _fit(
    vandermonde_func,
    input,
    other,
    degree,
    relative_condition=None,
    full=False,
    w=None,
):
    input = torch.tensor(input)
    other = torch.tensor(other)
    degree = torch.tensor(degree)

    if degree.ndim > 1:
        raise TypeError

    # if deg.dtype.kind not in "iu":
    #     raise TypeError

    if math.prod(degree.shape) == 0:
        raise TypeError

    if degree.min() < 0:
        raise ValueError

    if input.ndim != 1:
        raise TypeError

    if input.size == 0:
        raise TypeError

    if other.ndim < 1 or other.ndim > 2:
        raise TypeError

    if len(input) != len(other):
        raise TypeError

    if degree.ndim == 0:
        lmax = int(degree)
        van = vandermonde_func(input, lmax)
    else:
        degree, _ = torch.sort(degree)
        lmax = int(degree[-1])
        van = vandermonde_func(input, lmax)[:, degree]

    # set up the least squares matrices in transposed form
    lhs = van.T
    rhs = other.T

    if w is not None:
        w = torch.tensor(w)

        if w.ndim != 1:
            raise TypeError("expected 1D vector for w")

        if len(input) != len(w):
            raise TypeError("expected x and w to have same length")

        # apply weights. Don't use inplace operations as they
        # can cause problems with NA.
        lhs = lhs * w
        rhs = rhs * w

    # set rcond
    if relative_condition is None:
        relative_condition = len(input) * torch.finfo(input.dtype).eps

    # Determine the norms of the design matrix columns.
    if torch.is_complex(lhs):
        scl = torch.sqrt((torch.square(lhs.real) + torch.square(lhs.imag)).sum(1))
    else:
        scl = torch.sqrt(torch.square(lhs).sum(1))

    scl = torch.where(scl == 0, 1, scl)

    # Solve the least squares problem.
    c, resids, rank, s = torch.linalg.lstsq(lhs.T / scl, rhs.T, relative_condition)

    c = (c.T / scl).T

    # Expand c to include non-fitted coefficients which are set to zero
    if degree.ndim > 0:
        if c.ndim == 2:
            cc = torch.zeros((lmax + 1, c.shape[1]), dtype=c.dtype)
        else:
            cc = torch.zeros(lmax + 1, dtype=c.dtype)

        cc[degree] = c

        c = cc

    if full:
        return c, [resids, rank, s, relative_condition]
    else:
        return c


def _flattened_vandermonde(
    vandermonde_functions,
    points,
    degrees,
):
    vandermonde = _vandermonde(
        vandermonde_functions,
        points,
        degrees,
    )

    return torch.reshape(
        vandermonde,
        vandermonde.shape[: -len(degrees)] + (-1,),
    )


def _from_roots(
    f: Callable,
    g: Callable,
    input: Tensor,
) -> Tensor:
    if math.prod(input.shape) == 0:
        return torch.ones([1])

    input, _ = torch.sort(input)

    ys = []

    for x in input:
        a = torch.zeros(input.shape[0] + 1, dtype=x.dtype)
        b = f(-x, 1)

        [a, b] = _as_series([a, b])

        if a.shape[0] > b.shape[0]:
            y = torch.concatenate(
                [
                    b,
                    torch.zeros(
                        a.shape[0] - b.shape[0],
                        dtype=b.dtype,
                    ),
                ],
            )

            y = a + y
        else:
            y = torch.concatenate(
                [
                    a,
                    torch.zeros(
                        b.shape[0] - a.shape[0],
                        dtype=a.dtype,
                    ),
                ]
            )

            y = b + y

        ys = [*ys, y]

    p = torch.stack(ys)

    m = p.shape[0]

    x = m, p

    while x[0] > 1:
        m, r = divmod(x[0], 2)

        z = x[1]

        previous = torch.zeros([len(p), input.shape[0] + 1])

        y = previous

        for i in range(0, m):
            y[i] = g(z[i], z[i + m])[: input.shape[0] + 1]

        previous = y

        if r:
            previous[0] = g(previous[0], z[2 * m])[: input.shape[0] + 1]

        x = m, previous

    _, output = x

    return output[0]


def _get_domain(input: Tensor) -> Tensor:
    if torch.is_complex(input):
        output = torch.tensor(
            [
                torch.min(input.real) + 1j * torch.min(input.imag),
                torch.max(input.real) + 1j * torch.max(input.imag),
            ],
        )
        return output
    else:
        output = torch.tensor(
            [
                torch.min(input),
                torch.max(input),
            ],
        )

    return output


def _map_domain(x: Tensor, y: Tensor, z: Tensor) -> Tensor:
    (a, b), (c, d) = y, z

    return (b * c - a * d) / (b - a) + (d - c) / (b - a) * x


def _map_parameters(input: Tensor, other: Tensor) -> Tensor:
    a = input[1] - input[0]
    b = other[1] - other[0]

    x = (input[1] * other[0] - input[0] * other[1]) / a
    y = b / a

    return torch.tensor([x, y])


def _nonzero(
    input,
    size=1,
    fill_value=0,
):
    output = torch.nonzero(input, as_tuple=False)

    if output.shape[0] > size:
        output = output[:size]
    elif output.shape[0] < size:
        output = torch.concatenate(
            [
                output,
                torch.full(
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


def _normed_hermite_e_n(
    x: Tensor,
    n,
) -> Tensor:
    if n == 0:
        output = torch.full(x.shape, 1.0 / math.sqrt(math.sqrt(2.0 * math.pi)))
    else:
        a = torch.zeros_like(x)
        b = torch.ones_like(x) / math.sqrt(math.sqrt(2.0 * math.pi))

        size = torch.tensor(n)

        for _ in range(0, n - 1):
            previous = a

            a = -b * torch.sqrt((size - 1.0) / size)

            b = previous + b * x * torch.sqrt(1.0 / size)

            size = size - 1.0

        output = a + b * x

    return output


def _normed_hermite_n(
    x: Tensor,
    n,
) -> Tensor:
    if n == 0:
        output = torch.full(x.shape, 1 / math.sqrt(math.sqrt(math.pi)))
    else:
        a = torch.zeros_like(x)

        b = torch.ones_like(x) / math.sqrt(math.sqrt(math.pi))

        size = torch.tensor(n)

        for _ in range(0, n - 1):
            previous = a

            a = -b * torch.sqrt((size - 1.0) / size)

            b = previous + b * x * torch.sqrt(2.0 / size)

            size = size - 1.0

        output = a + b * x * math.sqrt(2.0)

    return output


def _nth_slice(
    i,
    ndim,
):
    sl = [None] * ndim
    sl[i] = slice(None)
    return tuple(sl)


def _pad_along_axis(
    input: Tensor,
    padding=(0, 0),
    axis=0,
):
    input = torch.moveaxis(input, axis, 0)

    if padding[0] < 0:
        input = input[torch.abs(padding[0]) :]

        padding = (0, padding[1])

    if padding[1] < 0:
        input = input[: -torch.abs(padding[1])]

        padding = (padding[0], 0)

    npad = torch.tensor([(0, 0)] * input.ndim)

    npad[0] = padding

    output = torch._numpy._funcs_impl.pad(
        input,
        pad_width=npad,
        mode="constant",
        constant_values=0,
    )

    return torch.moveaxis(output, 0, axis)


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
            output = torch.tensor([1], dtype=input.dtype)
        case 1:
            output = input
        case _:
            output = torch.zeros(input.shape[0] * exponent, dtype=input.dtype)

            [output, input] = _as_series([output, input])

            if output.shape[0] > input.shape[0]:
                input = torch.concatenate(
                    [
                        input,
                        torch.zeros(
                            output.shape[0] - input.shape[0],
                            dtype=input.dtype,
                        ),
                    ],
                )

                output = output + input
            else:
                output = torch.concatenate(
                    [
                        output,
                        torch.zeros(
                            input.shape[0] - output.shape[0],
                            dtype=output.dtype,
                        ),
                    ]
                )

                output = input + output

            for _ in range(2, _exponent + 1):
                output = func(output, input, mode="same")

    return output


def _trim_coefficients(
    input: Tensor,
    tol: float = 0.0,
) -> Tensor:
    if tol < 0:
        raise ValueError

    [input] = _as_series([input])

    indices = torch.nonzero(torch.abs(input) > tol)

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


def _vandermonde(functions, input: Tensor, degrees: Tensor) -> Tensor:
    n_dims = len(functions)

    if n_dims != len(input):
        raise ValueError

    if n_dims != len(degrees):
        raise ValueError

    if n_dims == 0:
        raise ValueError

    # produce the vandermonde matrix for each dimension, placing the last
    # axis of each in an independent trailing axis of the output
    vander_arrays = (
        functions[i](input[i], degrees[i])[(...,) + _nth_slice(i, n_dims)]
        for i in range(n_dims)
    )

    # we checked this wasn't empty already, so no `initial` needed
    return functools.reduce(operator.mul, vander_arrays)


def _z_series_mul(
    input: Tensor,
    other: Tensor,
    mode: Literal["full", "same", "valid"] = "full",
) -> Tensor:
    return torchaudio.functional.convolve(input, other, mode=mode)


def _z_series_to_c_series(
    input: Tensor,
) -> Tensor:
    n = (math.prod(input.shape) + 1) // 2

    c = input[n - 1 :]

    c[1:n] = c[1:n] * 2.0

    return c


def cheb2poly(input: Tensor) -> Tensor:
    [input] = _as_series([input])

    n = input.shape[0]

    if n < 3:
        return input

    c0 = torch.zeros_like(input)
    c0[0] = input[-2]

    c1 = torch.zeros_like(input)
    c1[0] = input[-1]

    for index in range(0, n - 2):
        i1 = n - 1 - index

        tmp = c0

        c0 = polysub(input[i1 - 2], c1)

        c1 = polyadd(tmp, polymulx(c1, "same") * 2)

    output = polymulx(c1, "same")

    output = polyadd(c0, output)

    return output


def chebder(
    input: Tensor,
    order=1,
    scale=1,
    axis=0,
) -> Tensor:
    if order < 0:
        raise ValueError

    [input] = _as_series([input])

    if order == 0:
        return input

    output = torch.moveaxis(input, axis, 0)

    n = output.shape[0]

    if order >= n:
        output = torch.zeros_like(output[:1])
    else:
        for _ in range(order):
            n = n - 1

            output = output * scale

            derivative = torch.empty((n,) + output.shape[1:], dtype=output.dtype)

            for i in range(0, n - 2):
                j = n - i

                derivative[j - 1] = (2 * j) * output[j]

                output = output.at[j - 2].add((j * output[j]) / (j - 2))

            if n > 1:
                derivative[1] = 4 * output[2]

            derivative[0] = output[1]

    return torch.moveaxis(output, 0, axis)


def chebdiv(
    input: Tensor,
    other: Tensor,
) -> Tuple[Tensor, Tensor]:
    return _div(chebmul, input, other)


def chebfit(
    input: Tensor,
    other: Tensor,
    degree: Tensor | int,
    relative_condition: float | None = None,
    full: bool = False,
    weight: Tensor | None = None,
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


def chebfromroots(
    input: Tensor,
) -> Tensor:
    return _from_roots(chebline, chebmul, input)


def chebgauss(
    degree: int,
) -> Tuple[Tensor, Tensor]:
    if not degree > 0:
        raise ValueError

    output = torch.cos(torch.arange(1, 2 * degree, 2) / (2 * degree) * math.pi)

    weight = torch.ones(degree) * (math.pi / degree)

    return output, weight


def chebgrid2d(
    x: Tensor,
    y: Tensor,
    c: Tensor,
) -> Tensor:
    for arg in [x, y]:
        c = chebval(arg, c)
    return c


def chebgrid3d(
    x: Tensor,
    y: Tensor,
    z: Tensor,
    c: Tensor,
) -> Tensor:
    for arg in [x, y, z]:
        c = chebval(arg, c)
    return c


def chebint(
    c: Tensor,
    order=1,
    k=None,
    lower_bound=0,
    scale=1,
    axis=0,
) -> Tensor:
    if k is None:
        k = []

    [c] = _as_series([c])

    lower_bound = torch.tensor(lower_bound)

    scale = torch.tensor(scale)

    if not numpy.iterable(k):
        k = [k]

    if len(k) > order:
        raise ValueError

    if lower_bound.ndim != 0:
        raise ValueError

    if scale.ndim != 0:
        raise ValueError

    if order == 0:
        return c

    c = torch.moveaxis(c, axis, 0)

    k = torch.tensor([*k] + [0.0] * (order - len(k)))

    k = torch.atleast_1d(k)

    for i in range(order):
        n = c.shape[0]

        c = c * scale

        tmp = torch.empty([n + 1, *c.shape[1:]])

        tmp[0] = c[0] * 0
        tmp[1] = c[0]

        if n > 1:
            tmp[2] = c[1] / 4

        if n < 2:
            j = torch.tensor([], dtype=torch.int32)
        else:
            j = torch.arange(2, n)

        tmp[j + 1] = (c[j].T / (2 * (j + 1))).T
        tmp[j - 1] = tmp[j - 1] + -(c[j] / (2 * (j - 1)))

        tmp[0] = tmp[0] + (k[i] - chebval(lower_bound, tmp))

        c = tmp

    c = torch.moveaxis(c, 0, axis)

    return c


def chebinterpolate(
    func,
    degree,
    args=(),
):
    _deg = int(degree)

    if _deg != degree:
        raise ValueError

    if _deg < 0:
        raise ValueError

    order = _deg + 1
    xcheb = chebpts1(order)

    yfunc = func(xcheb, *args)

    m = chebvander(xcheb, _deg)

    c = m.T @ yfunc

    c[0] /= order

    c[1:] /= 0.5 * order

    return c


def chebline(
    input: float,
    other: float,
) -> Tensor:
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

    output = torch.zeros(input.shape[0] + 1, dtype=input.dtype)

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

    match _exponent:
        case 0:
            output = torch.tensor([1.0], dtype=input.dtype)
        case 1:
            output = input
        case _:
            output = torch.zeros(input.shape[0] * exponent, dtype=input.dtype)

            output = chebadd(output, input)

            zs = _c_series_to_z_series(input)

            output = _c_series_to_z_series(output)

            for _ in range(2, _exponent + 1):
                output = torchaudio.functional.convolve(output, zs, mode="same")

            output = _z_series_to_c_series(output)

    return output


def chebpts1(
    input: int,
) -> Tensor:
    if input < 1:
        raise ValueError

    return torch.sin(0.5 * math.pi / input * torch.arange(-input + 1, input + 1, 2))


def chebpts2(
    input: int,
) -> Tensor:
    if input < 2:
        raise ValueError

    return torch.cos(torch.linspace(-math.pi, 0, input))


def chebroots(input: Tensor) -> Tensor:
    [input] = _as_series([input])

    if input.shape[0] <= 1:
        return torch.tensor([], dtype=input.dtype)

    if input.shape[0] == 2:
        return torch.tensor([-input[0] / input[1]])

    output = chebcompanion(input)

    output = torch.flip(output, dims=[0])
    output = torch.flip(output, dims=[1])

    output = torch.linalg.eigvals(output)

    output, _ = torch.sort(output.real)

    return output


def chebval(
    input: Tensor,
    coefficients: Tensor,
    tensor: bool = True,
) -> Tensor:
    [coefficients] = _as_series([coefficients])

    if tensor:
        coefficients = torch.reshape(
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
            a = coefficients[-2] * torch.ones_like(input)
            b = coefficients[-1] * torch.ones_like(input)

            for i in range(3, coefficients.shape[0] + 1):
                previous = a

                a = coefficients[-i] - b
                b = previous + b * 2.0 * input

    return a + b * input


def chebval2d(
    x: Tensor,
    y: Tensor,
    c: Tensor,
) -> Tensor:
    return _evaluate(chebval, c, x, y)


def chebval3d(
    x: Tensor,
    y: Tensor,
    z: Tensor,
    c: Tensor,
) -> Tensor:
    return _evaluate(chebval, c, x, y, z)


def chebvander(
    x: Tensor,
    degree: Tensor,
) -> Tensor:
    if degree < 0:
        raise ValueError

    x = torch.atleast_1d(x)
    dims = (degree + 1,) + x.shape
    dtyp = torch.promote_types(x.dtype, torch.tensor(0.0).dtype)
    x = x.to(dtyp)
    v = torch.empty(dims, dtype=dtyp)

    v[0] = torch.ones_like(x)

    if degree > 0:
        v[1] = x

        x2 = 2 * x

        for index in range(2, degree + 1):
            v[index] = v[index - 1] * x2 - v[index - 2]

    return torch.moveaxis(v, 0, -1)


def chebvander2d(
    x: Tensor,
    y: Tensor,
    degree: Tensor,
) -> Tensor:
    return _flattened_vandermonde(
        (chebvander, chebvander),
        (x, y),
        degree,
    )


def chebvander3d(
    x: Tensor,
    y: Tensor,
    z: Tensor,
    degree: Tensor,
) -> Tensor:
    return _flattened_vandermonde(
        (chebvander, chebvander, chebvander),
        (x, y, z),
        degree,
    )


def chebweight(
    input: Tensor,
) -> Tensor:
    return 1.0 / (torch.sqrt(1.0 + input) * torch.sqrt(1.0 - input))


def herm2poly(
    c: Tensor,
) -> Tensor:
    [c] = _as_series([c])
    n = c.shape[0]

    if n == 1:
        return c

    if n == 2:
        c[1] = c[1] * 2

        return c
    else:
        c0 = torch.zeros_like(c)
        c0[0] = c[-2]

        c1 = torch.zeros_like(c)
        c1[0] = c[-1]

        def body(k, c0c1):
            i = n - 1 - k
            c0, c1 = c0c1
            tmp = c0
            c0 = polysub(c[i - 2], c1 * (2 * (i - 1)))
            c1 = polyadd(tmp, polymulx(c1, "same") * 2)
            return c0, c1

        x = (c0, c1)

        y = x

        for index in range(0, n - 2):
            y = body(index, y)

        c0, c1 = y

        return polyadd(c0, polymulx(c1, "same") * 2)


def hermder(
    c,
    order=1,
    scale=1,
    axis=0,
) -> Tensor:
    if order < 0:
        raise ValueError

    [c] = _as_series([c])

    if order == 0:
        return c

    c = torch.moveaxis(c, axis, 0)

    n = c.shape[0]

    if order >= n:
        c = torch.zeros_like(c[:1])
    else:
        for _ in range(order):
            n = n - 1

            c *= scale

            der = torch.empty((n,) + c.shape[1:], dtype=c.dtype)

            j = torch.arange(n, 0, -1)

            der[j - 1] = (2 * j * (c[j]).T).T

            c = der

    c = torch.moveaxis(c, 0, axis)

    return c


def hermdiv(
    input: Tensor,
    other: Tensor,
) -> Tuple[Tensor, Tensor]:
    return _div(hermmul, input, other)


def herme2poly(c: Tensor) -> Tensor:
    [c] = _as_series([c])

    n = c.shape[0]

    if n == 1:
        return c

    if n == 2:
        return c
    else:
        c0 = torch.zeros_like(c)
        c0[0] = c[-2]

        c1 = torch.zeros_like(c)
        c1[0] = c[-1]

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


def hermeder(
    c,
    order=1,
    scale=1,
    axis=0,
) -> Tensor:
    if order < 0:
        raise ValueError

    [c] = _as_series([c])

    if order == 0:
        return c

    c = torch.moveaxis(c, axis, 0)

    n = c.shape[0]

    if order >= n:
        c = torch.zeros_like(c[:1])
    else:
        for _ in range(order):
            n = n - 1

            c = c * scale

            der = torch.empty((n,) + c.shape[1:], dtype=c.dtype)

            j = torch.arange(n, 0, -1)

            der[j - 1] = (j * (c[j]).T).T

            c = der

    c = torch.moveaxis(c, 0, axis)

    return c


def hermediv(
    input: Tensor,
    other: Tensor,
) -> Tuple[Tensor, Tensor]:
    return _div(hermemul, input, other)


def hermefit(
    input: Tensor,
    other: Tensor,
    degree: Tensor | int,
    relative_condition: float | None = None,
    full: bool = False,
    weight: Tensor | None = None,
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


chebtrim = _trim_coefficients

hermetrim = _trim_coefficients

hermtrim = _trim_coefficients

lagtrim = _trim_coefficients

legtrim = _trim_coefficients

polytrim = _trim_coefficients

__all__ = [
    "_c_series_to_z_series",
    "_fit",
    "_get_domain",
    "_map_domain",
    "_map_parameters",
    "_pow",
    "_trim_coefficients",
    "_trim_sequence",
    "_vandermonde",
    "_z_series_to_c_series",
    "cheb2poly",
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
    "herm2poly",
    "hermder",
    "hermdiv",
    "hermdomain",
    "herme2poly",
    "hermeder",
    "hermediv",
    "hermedomain",
    "hermefit",
    "hermeone",
    "hermetrim",
    "hermex",
    "hermezero",
    "hermone",
    "hermtrim",
    "hermx",
    "hermzero",
    "lagdomain",
    "lagone",
    "lagtrim",
    "lagx",
    "lagzero",
    "legdomain",
    "legone",
    "legtrim",
    "legx",
    "legzero",
    "polydomain",
    "polyone",
    "polytrim",
    "polyx",
    "polyzero",
]
