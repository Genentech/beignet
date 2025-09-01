import math

import torch
from torch import Tensor


def chebyshev_gauss_quadrature(degree: int) -> tuple[Tensor, Tensor]:
    if not degree > 0:
        raise ValueError

    output = torch.cos(torch.arange(1, 2 * degree, 2) / (2 * degree) * math.pi)

    weight = torch.ones(degree) * (math.pi / degree)

    return output, weight
