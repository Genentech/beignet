import torch
from torch import Tensor

from .__particles_per_cell import _particles_per_cell


def _estimate_cell_capacity(
        positions: Tensor,
        size: Tensor,
        unit_size: float,
        buffer_size_multiplier: float
) -> int:
    cell_capacity = torch.max(_particles_per_cell(positions, size, unit_size))

    return int(cell_capacity * buffer_size_multiplier)
