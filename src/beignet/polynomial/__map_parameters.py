import torch
from torch import Tensor


def _map_parameters(input: Tensor, other: Tensor) -> (Tensor, Tensor):
    oldlen = input[1] - input[0]
    newlen = other[1] - other[0]
    off = (input[1] * other[0] - input[0] * other[1]) / oldlen
    scale = newlen / oldlen

    off = torch.ravel(off)
    scale = torch.ravel(scale)

    return torch.concatenate([off, scale])
