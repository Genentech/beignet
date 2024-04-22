import torch
from torch import Tensor


def _is_space_valid(space: Tensor) -> bool:
    if space.ndim == 0 or space.ndim == 1:
        return torch.tensor([True])

    if space.ndim == 2:
        return torch.tensor([torch.all(torch.triu(space) == space)])

    return torch.tensor([False])
