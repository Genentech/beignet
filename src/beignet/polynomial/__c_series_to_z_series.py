import math

import torch
from torch import Tensor


def _c_series_to_z_series(
    input: Tensor,
) -> Tensor:
    index = math.prod(input.shape)

    zs = torch.zeros(2 * index - 1, dtype=input.dtype)

    zs[index - 1 :] = input / 2.0

    return torch.flip(zs, dims=[0]) + zs
