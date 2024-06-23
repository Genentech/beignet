import torch
from torch import Tensor


def laguerre_series_weight(x: Tensor) -> Tensor:
    return torch.exp(-x)
