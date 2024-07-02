from typing import Literal

import torch
from torch import Tensor

from .__as_series import _as_series


def polymulx(
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
    [input] = _as_series([input])

    output = torch.zeros(input.shape[0] + 1, dtype=input.dtype)

    output[1:] = input

    if mode == "same":
        output = output[: input.shape[0]]

    return output
