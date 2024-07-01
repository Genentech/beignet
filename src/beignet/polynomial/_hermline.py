import torch
from torch import Tensor


def hermline(
    input: float,
    other: float,
) -> Tensor:
    return torch.tensor([input, other / 2])
