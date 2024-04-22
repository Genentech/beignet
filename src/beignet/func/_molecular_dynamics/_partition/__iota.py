import torch
from torch import Tensor


def _iota(shape: tuple[int, ...], dim: int = 0, **kwargs) -> Tensor:
    dimensions = []

    for index, _ in enumerate(shape):
        if index != dim:
            dimension = 1
        else:
            dimension = shape[index]

        dimensions = [*dimensions, dimension]

    return torch.arange(shape[dim], **kwargs).view(*dimensions).expand(*shape)
