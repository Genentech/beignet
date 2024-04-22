import torch
from torch import Tensor


def _cell_size(box: Tensor, minimum_unit_size: Tensor) -> Tensor:
    return box / torch.floor(box / minimum_unit_size)
