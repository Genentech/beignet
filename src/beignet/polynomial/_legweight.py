import torch
from torch import Tensor


def legweight(x: Tensor) -> Tensor:
    return torch.ones_like(x)
