import torch
from torch import Tensor


def physicists_hermite_series_weight(x: Tensor) -> Tensor:
    return torch.exp(-(x**2))
