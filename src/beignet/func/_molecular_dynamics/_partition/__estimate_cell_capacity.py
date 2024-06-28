import torch
from torch import Tensor

from .__particles_per_cell import _particles_per_cell


def _estimate_cell_capacity(
    positions: Tensor, size: Tensor, unit_size: float, buffer_size_multiplier: float
) -> int:
    r"""Estimates the capacity of a cell based on particle positions, cell size, unit size, and a buffer size multiplier.

    Parameters
    ----------
    positions : Tensor
        A tensor containing the positions of particles.
    size : Tensor
        A tensor representing the size of the cell.
    unit_size : float
        The size of a single unit within the cell.
    buffer_size_multiplier : float
        A multiplier to account for buffer space in the cell capacity.

    Returns
    -------
    int
        The estimated capacity of the cell, adjusted by the buffer size multiplier.
    """
    cell_capacity = torch.max(_particles_per_cell(positions, size, unit_size))

    return int(cell_capacity * buffer_size_multiplier)
