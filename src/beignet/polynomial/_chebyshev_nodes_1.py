import torch
from torch import Tensor


def chebyshev_nodes_1(input: Tensor) -> Tensor:
    points = int(input)

    if points != input:
        raise ValueError

    if points < 1:
        raise ValueError

    output = 0.5 * torch.pi / points * torch.arange(-points + 1, points + 1, 2)

    return torch.sin(output)
