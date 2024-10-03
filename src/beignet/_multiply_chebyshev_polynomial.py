import math
from typing import Literal

import torch
from torch import Tensor
from ._convolve import convolve


def multiply_chebyshev_polynomial(
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

    index = math.prod(input.shape)
    output1 = torch.zeros(2 * index - 1, dtype=input.dtype)
    output1[index - 1 :] = input / 2.0
    output1 = torch.flip(output1, dims=[0]) + output1
    a = output1

    index1 = math.prod(other.shape)
    output2 = torch.zeros(2 * index1 - 1, dtype=other.dtype)
    output2[index1 - 1 :] = other / 2.0
    output2 = torch.flip(output2, dims=[0]) + output2
    b = output2

    output = convolve(a, b, mode=mode)

    n = (math.prod(output.shape) + 1) // 2
    c = output[n - 1 :]
    c[1:n] = c[1:n] * 2.0
    output = c

    if mode == "same":
        output = output[: max(input.shape[0], other.shape[0])]

    return output
