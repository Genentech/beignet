import math

import torch
from torch import Tensor


def _hash_constants(spatial_dimensions: int, cells_per_side: Tensor) -> Tensor:
    if cells_per_side.ndim == 0:
        constants = []

        for spatial_dimension in range(spatial_dimensions):
            constants = [
                *constants,
                math.pow(cells_per_side, spatial_dimension),
            ]

        return torch.tensor([constants], dtype=torch.int32)

    if cells_per_side.size == spatial_dimensions:
        cells_per_side = torch.concatenate(
            [
                torch.tensor([[1]], dtype=torch.int32),
                cells_per_side[:, :-1],
            ],
            dim=1,
        )

        return torch.cumprod(torch.flatten(cells_per_side), dim=0)

    raise ValueError
