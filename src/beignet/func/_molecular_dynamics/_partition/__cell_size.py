import torch
from torch import Tensor


def _cell_size(box: Tensor, minimum_unit_size: Tensor) -> Tensor:
    if box.shape == minimum_unit_size.shape or minimum_unit_size.ndim == 0:
        return box / torch.floor(box / minimum_unit_size)

    else:
        raise ValueError("Box and minimum unit size must be of the same shape.")
