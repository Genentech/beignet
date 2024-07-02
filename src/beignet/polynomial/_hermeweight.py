import torch
from torch import Tensor


def hermeweight(x: Tensor) -> Tensor:
    return torch.exp(-0.5 * x**2)
