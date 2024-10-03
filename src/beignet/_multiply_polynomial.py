from typing import Literal

import torch
from torch import Tensor
from ._convolve import convolve


def multiply_polynomial(
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

    output = convolve(input, other)

    if mode == "same":
        output = output[: max(input.shape[0], other.shape[0])]

    return output
