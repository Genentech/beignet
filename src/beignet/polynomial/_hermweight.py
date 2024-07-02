import torch
from torch import Tensor


def hermweight(x: Tensor) -> Tensor:
    return torch.exp(-(x**2))
