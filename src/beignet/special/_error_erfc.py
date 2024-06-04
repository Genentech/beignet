import torch
from torch import Tensor

from ._faddeeva_w import faddeeva_w


def error_erfc(input: Tensor) -> Tensor:
    r"""
    Complementary error function.

    Parameters
    ----------
    input : Tensor

    Returns
    -------
    Tensor
    """
    return torch.exp(-(input**2)) * faddeeva_w(1.0j * input)
