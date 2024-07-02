import torch
from torch import Tensor


def legline(
    input: float,
    other: float,
) -> Tensor:
    return torch.tensor([input, other])
