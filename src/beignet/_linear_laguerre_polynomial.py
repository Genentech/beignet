import torch
from torch import Tensor


def linear_laguerre_polynomial(input: float, other: float) -> Tensor:
    return torch.tensor([input + other, -other])
