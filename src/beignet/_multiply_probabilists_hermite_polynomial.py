from typing import Literal

import torch
from torch import Tensor

from ._add_probabilists_hermite_polynomial import add_probabilists_hermite_polynomial
from ._multiply_probabilists_hermite_polynomial_by_x import (
    multiply_probabilists_hermite_polynomial_by_x,
)
from ._subtract_probabilists_hermite_polynomial import (
    subtract_probabilists_hermite_polynomial,
)


def multiply_probabilists_hermite_polynomial(
    input: Tensor,
    other: Tensor,
    mode: Literal["full", "same", "valid"] = "full",
) -> Tensor:
    r"""
    Returns the product of two polynomials.

    Parameters
    ----------
    input : Tensor
        Polynomial coefficients.

    other : Tensor
        Polynomial coefficients.

    Returns
    -------
    output : Tensor
        Polynomial coefficients of the product.
    """
    input = torch.atleast_1d(input)
    other = torch.atleast_1d(other)

    dtype = torch.promote_types(input.dtype, other.dtype)

    input = input.to(dtype)
    other = other.to(dtype)

    m, n = input.shape[0], other.shape[0]

    if m > n:
        x, y = other, input
    else:
        x, y = input, other

    match x.shape[0]:
        case 1:
            a = add_probabilists_hermite_polynomial(torch.zeros(m + n - 1), x[0] * y)
            b = torch.zeros(m + n - 1)
        case 2:
            a = add_probabilists_hermite_polynomial(torch.zeros(m + n - 1), x[0] * y)
            b = add_probabilists_hermite_polynomial(torch.zeros(m + n - 1), x[1] * y)
        case _:
            size = x.shape[0]

            a = add_probabilists_hermite_polynomial(torch.zeros(m + n - 1), x[-2] * y)
            b = add_probabilists_hermite_polynomial(torch.zeros(m + n - 1), x[-1] * y)

            for i in range(3, x.shape[0] + 1):
                previous = a

                size = size - 1

                a = subtract_probabilists_hermite_polynomial(
                    x[-i] * y, b * (size - 1.0)
                )

                b = add_probabilists_hermite_polynomial(
                    previous, multiply_probabilists_hermite_polynomial_by_x(b, "same")
                )

    output = add_probabilists_hermite_polynomial(
        a, multiply_probabilists_hermite_polynomial_by_x(b, "same")
    )

    if mode == "same":
        output = output[: max(m, n)]

    return output
