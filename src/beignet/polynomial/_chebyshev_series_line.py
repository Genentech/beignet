import torch
from torch import Tensor


def chebyshev_series_line(input: Tensor, other: Tensor) -> Tensor:
    if other == 0:
        return torch.tensor([input])
    else:
        return torch.tensor([input, other])
