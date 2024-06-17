import torch
from torch import Tensor


def laguerre_polynomial_weight(x: Tensor) -> Tensor:
    return torch.exp(-x)
