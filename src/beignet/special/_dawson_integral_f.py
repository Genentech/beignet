import math

import torch
from torch import Tensor

import beignet.special


def dawson_integral_f(input: Tensor) -> Tensor:
    return (
        math.sqrt(torch.pi)
        / 2.0
        * torch.exp(-torch.square(input))
        * beignet.special.error_erfi(input)
    )
