import functools
import math
import operator

import torch
from torch import Tensor


def _cell_dimensions(
    spatial_dimension: int,
    box_size: Tensor,
    minimum_cell_size: float,
) -> (Tensor, Tensor, Tensor, int):
    r"""Compute the number of cells-per-side and total number of cells in a box.

    Parameters:
    -----------
    spatial_dimension : int
        The spatial dimension of the box (e.g., 2 for 2D, 3 for 3D).
    box_size : Tensor or scalar
        The size of the box. Can be a scalar or a Tensor with dimensions 0, 1, or 2.
    minimum_cell_size : float
        The minimum size of the cells.

    Returns:
    --------
    box_size : Tensor
        The (possibly modified) size of the box.
    cell_size : Tensor
        The size of the cells.
    cells_per_side : Tensor
        The number of cells per side.
    cell_count : int
        The total number of cells in the box.

    Raises:
    -------
    ValueError
        If the box size is less than the minimum cell size.
        If the box is not at least 3x the size of the grid spacing in each dimension.
        If the box is not a scalar, a vector, or a matrix.
    """
    if isinstance(box_size, (int, float)):
        box_size = float(box_size)

        if box_size < minimum_cell_size:
            raise ValueError("Box size must be at least as large as minimum cell size.")

    if isinstance(box_size, Tensor):
        if box_size.dtype in {torch.int32, torch.int64}:
            box_size = box_size.float()

    if isinstance(box_size, Tensor):
        cells_per_side = torch.floor(box_size / minimum_cell_size)

        cell_size = box_size / cells_per_side

        cells_per_side = cells_per_side.to(torch.int32)

        if box_size.dim() == 1 or box_size.dim() == 2:
            assert box_size.numel() == spatial_dimension

            flattened_cells_per_side = cells_per_side.view(-1)

            for cells in flattened_cells_per_side:
                if cells.item() < 3:
                    raise ValueError(
                        "Box must be at least 3x the size of the grid spacing in each dimension."
                    )

            cell_count = functools.reduce(
                operator.mul,
                flattened_cells_per_side,
                1,
            )

        elif box_size.dim() == 0:
            cell_count = cells_per_side**spatial_dimension

        else:
            raise ValueError(
                f"Box must be either: a scalar, a vector, or a matrix. Found {box_size}."
            )

    else:
        cells_per_side = math.floor(box_size / minimum_cell_size)

        cell_size = box_size / cells_per_side

        cell_count = cells_per_side**spatial_dimension

    return box_size, cell_size, cells_per_side, int(cell_count)
