import torch
from torch import Tensor


def _map_parameters(input: Tensor, other: Tensor) -> Tensor:
    a = input[1] - input[0]
    b = other[1] - other[0]

    x = (input[1] * other[0] - input[0] * other[1]) / a
    y = b / a

    return torch.tensor([x, y])
