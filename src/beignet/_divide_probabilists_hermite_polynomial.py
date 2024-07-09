from typing import Tuple

import torch
from torch import Tensor

from ._multiply_probabilists_hermite_polynomial import (
    multiply_probabilists_hermite_polynomial,
)


def divide_probabilists_hermite_polynomial(
    input: Tensor,
    other: Tensor,
) -> Tuple[Tensor, Tensor]:
    r"""
    Returns the quotient and remainder of two polynomials.

    Parameters
    ----------
    input : Tensor
        Polynomial coefficients.

    other : Tensor
        Polynomial coefficients.

    Returns
    -------
    output : Tuple[Tensor, Tensor]
        Polynomial coefficients of the quotient and remainder.
    """
    input = torch.atleast_1d(input)
    other = torch.atleast_1d(other)

    dtype = torch.promote_types(input.dtype, other.dtype)

    input = input.to(dtype)
    other = other.to(dtype)

    m = input.shape[0]
    n = other.shape[0]

    if m < n:
        return torch.zeros_like(input[:1]), input

    if n == 1:
        return input / other[-1], torch.zeros_like(input[:1])

    def f(x: Tensor) -> Tensor:
        indicies = torch.flip(x, [0])

        indicies = torch.nonzero(indicies, as_tuple=False)

        if indicies.shape[0] > 1:
            indicies = indicies[:1]

        if indicies.shape[0] < 1:
            indicies = torch.concatenate(
                [
                    indicies,
                    torch.full(
                        [
                            1 - indicies.shape[0],
                            indicies.shape[1],
                        ],
                        0,
                    ),
                ],
                0,
            )

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

        p = multiply_probabilists_hermite_polynomial(y2, other)

        pidx = f(p)

        t = remainder[ridx1] / p[pidx]

        remainder_modified = remainder.clone()
        remainder_modified[ridx1] = 0.0

        a = remainder_modified

        p_modified = p.clone()
        p_modified[pidx] = 0.0

        b = t * p_modified

        a = torch.atleast_1d(a)
        b = torch.atleast_1d(b)

        dtype = torch.promote_types(a.dtype, b.dtype)

        a = a.to(dtype)
        b = b.to(dtype)

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
