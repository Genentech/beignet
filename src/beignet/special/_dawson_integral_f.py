import math

import torch
from torch import Tensor

from ._error_erfi import error_erfi


def dawson_integral_f(input: Tensor, *, out: Tensor | None = None) -> Tensor:
    r"""
    Dawson’s integral.

    Parameters
    ----------
    input : Tensor
        Input tensor.

    out : Tensor, optional
        Output tensor.

    Returns
    -------
    Tensor
    """
    output = math.sqrt(torch.pi) / 2.0 * torch.exp(-(input**2)) * error_erfi(input)

    if out is not None:
        out.copy_(output)

        return out

    return output
