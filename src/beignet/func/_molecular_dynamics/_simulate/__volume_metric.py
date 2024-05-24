import torch
from torch import Tensor


def _volume_metric(dimension: int, box: Tensor) -> Tensor:
    if torch.tensor(box).shape == torch.Size([]) or not box.ndim:
        return box**dimension

    match box.ndim:
        case 1:
            return torch.prod(box)
        case 2:
            return torch.linalg.det(box)
        case _:
            raise ValueError
