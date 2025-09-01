import math

import torch
from torch import Tensor


def chebyshev_extrema(input: int) -> Tensor:
    if input < 2:
        raise ValueError

    return torch.cos(torch.linspace(-math.pi, 0, input))
