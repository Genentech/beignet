from typing import Literal

import torch
from torch import Tensor


def multiply_polynomial_by_x(
    input: Tensor,
    mode: Literal["full", "same", "valid"] = "full",
) -> Tensor:
    r"""
    Parameters
    ----------
    input : Tensor
        Polynomial coefficients.

    mode : Literal["full", "same", "valid"]

    Returns
    -------
    output : Tensor
        Polynomial coefficients of the product of the polynomial and the
        independent variable.
    """
    input = torch.atleast_1d(input)

    output = torch.zeros(input.shape[0] + 1, dtype=input.dtype)

    output[1:] = input

    if mode == "same":
        output = output[: input.shape[0]]

    return output
