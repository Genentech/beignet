import torch
from torch import Tensor


def lagline(
    input: float,
    other: float,
) -> Tensor:
    return torch.tensor([input + other, -other])
