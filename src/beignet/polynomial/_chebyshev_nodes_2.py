import torch
from torch import Tensor


def chebyshev_nodes_2(input: Tensor) -> Tensor:
    points = int(input)

    if points != input:
        raise ValueError

    if points < 2:
        raise ValueError

    output = torch.linspace(-torch.pi, 0, points)

    return torch.cos(output)
