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
    if isinstance(box_size, int):
        box_size = float(box_size)

    if isinstance(box_size, Tensor):
        if box_size.dtype in {torch.int32, torch.int64}:
            box_size = float(box_size)

    cells_per_side = math.floor(box_size / minimum_cell_size)

    cell_size = box_size / cells_per_side

    cells_per_side = torch.tensor(cells_per_side, dtype=torch.int32)

    if isinstance(box_size, Tensor):
        if box_size.ndim == 1 or box_size.ndim == 2:
            assert box_size.size == spatial_dimension

            flattened_cells_per_side = torch.reshape(
                cells_per_side,
                [
                    -1,
                ],
            )

            for cells in flattened_cells_per_side:
                if cells < 3:
                    raise ValueError

            cell_count = functools.reduce(
                operator.mul,
                flattened_cells_per_side,
                1,
            )
        elif box_size.ndim == 0:
            cell_count = math.pow(cells_per_side, spatial_dimension)
        else:
            raise ValueError
    else:
        cell_count = math.pow(cells_per_side, spatial_dimension)

    return box_size, cell_size, cells_per_side, int(cell_count)
