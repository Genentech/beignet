import torch
from torch import Tensor


def hermeline(
    input: float,
    other: float,
) -> Tensor:
    return torch.tensor([input, other])
