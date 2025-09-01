import math

import torch
from torch import Tensor


def chebyshev_zeros(input: int) -> Tensor:
    if input < 1:
        raise ValueError

    return torch.sin(0.5 * math.pi / input * torch.arange(-input + 1, input + 1, 2))
