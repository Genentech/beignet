import torch
from torch import Tensor


def probabilists_hermite_polynomial_weight(x: Tensor) -> Tensor:
    return torch.exp(-0.5 * x**2)
