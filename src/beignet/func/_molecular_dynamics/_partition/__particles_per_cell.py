import torch
from torch import Tensor

from .__cell_dimensions import _cell_dimensions
from .__hash_constants import _hash_constants
from .__segment_sum import _segment_sum


def _particles_per_cell(
    positions: Tensor,
    size: Tensor,
    minimum_size: float,
) -> Tensor:
    """
    Computes the number of particles per cell given a defined cell size and minimum size.

    Parameters
    ----------
    positions : Tensor
        A tensor representing the positions of the particles in the system.
        The shape of the tensor is expected to be (n, d) where `n` is the number
        of particles and `d` is the dimensionality of the system.

    size : Tensor
        A tensor that defines the size of the space in each dimension.
        It should have the same shape as a single particle position.

    minimum_size : float
        A scalar that defines the minimum size of the cells.
        All cells will be at least this size in each dimension.

    Returns
    -------
    Tensor
        A tensor with the number of particles per cell. Each position in the tensor
        corresponds to a cell in the grid defined by the `size` and `minimum_size`
        parameters. The value at each position is the count of particles in that cell.
    """
    size, unit_size, per_side, n = _cell_dimensions(
        positions.shape[1], size, minimum_size
    )

    indexes = torch.sum(
        (positions / unit_size).to(dtype=torch.int32)
        * _hash_constants(positions.shape[1], per_side).to(device=positions.device),
        dim=1,
    )

    return _segment_sum(torch.ones_like(indexes), indexes, n)
