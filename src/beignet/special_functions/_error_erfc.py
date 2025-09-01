import torch
from torch import Tensor

from ._faddeeva_w import faddeeva_w


def error_erfc(input: Tensor, *, out: Tensor | None = None) -> Tensor:
    r"""
    Complementary error function.

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
    output = torch.exp(-(input**2)) * faddeeva_w(1.0j * input)

    if out is not None:
        out.copy_(output)

        return out

    return output
