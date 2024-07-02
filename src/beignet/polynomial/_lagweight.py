import torch
from torch import Tensor


def lagweight(x: Tensor) -> Tensor:
    return torch.exp(-x)
