import torch
from torch import Tensor

from ._faddeeva_w import faddeeva_w


def error_erfc(input: Tensor) -> Tensor:
    return torch.exp(-torch.pow(input, 2)) * faddeeva_w(1.0j * input)
