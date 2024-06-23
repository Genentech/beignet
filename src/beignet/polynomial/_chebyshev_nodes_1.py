import torch
from torch import Tensor


def chebyshev_nodes_1(input: Tensor) -> Tensor:
    _npts = int(input)

    if _npts != input:
        raise ValueError

    if _npts < 1:
        raise ValueError

    x = 0.5 * torch.pi / _npts * torch.arange(-_npts + 1, _npts + 1, 2)

    return torch.sin(x)
