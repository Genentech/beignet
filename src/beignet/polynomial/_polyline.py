import torch
from torch import Tensor


def polyline(input: float, other: float) -> Tensor:
    return torch.tensor([input, other])
