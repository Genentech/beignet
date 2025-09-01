import torch
from torch import Tensor


def linear_physicists_hermite_polynomial(input: float, other: float) -> Tensor:
    return torch.tensor([input, other / 2])
