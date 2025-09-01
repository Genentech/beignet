import torch
from torch import Tensor


def linear_probabilists_hermite_polynomial(input: float, other: float) -> Tensor:
    return torch.tensor([input, other])
