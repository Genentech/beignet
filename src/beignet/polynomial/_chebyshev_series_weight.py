import torch
from torch import Tensor


def chebyshev_series_weight(x: Tensor) -> Tensor:
    return 1.0 / (torch.sqrt(1.0 + x) * torch.sqrt(1.0 - x))
