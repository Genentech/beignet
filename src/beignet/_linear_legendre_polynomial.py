import torch
from torch import Tensor


def linear_legendre_polynomial(input: float, other: float) -> Tensor:
    return torch.tensor([input, other])
