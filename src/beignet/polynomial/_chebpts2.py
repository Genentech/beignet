import math

import torch
from torch import Tensor


def chebpts2(
    input: int,
) -> Tensor:
    if input < 2:
        raise ValueError

    return torch.cos(torch.linspace(-math.pi, 0, input))
