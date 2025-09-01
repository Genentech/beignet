import torch
from torch import Tensor


def legendre_polynomial_weight(x: Tensor) -> Tensor:
    return torch.ones_like(x)
