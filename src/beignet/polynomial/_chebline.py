import torch
from torch import Tensor


def chebline(
    input: float,
    other: float,
) -> Tensor:
    return torch.tensor([input, other])
