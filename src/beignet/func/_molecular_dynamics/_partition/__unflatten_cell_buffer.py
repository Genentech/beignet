import torch
from torch import Tensor


def _unflatten_cell_buffer(
    buffer: Tensor,
    cells_per_side: Tensor,
    dim: int,
) -> Tensor:
    if (
        isinstance(cells_per_side, int)
        or isinstance(cells_per_side, float)
        or (isinstance(cells_per_side, Tensor) and not cells_per_side.shape)
    ):
        cells_per_side = (int(cells_per_side),) * dim
    elif isinstance(cells_per_side, Tensor) and len(cells_per_side.shape) == 1:
        cells_per_side = [int(units) for units in cells_per_side[::-1]]

        cells_per_side = tuple(cells_per_side)
    elif isinstance(cells_per_side, Tensor) and len(cells_per_side.shape) == 2:
        cells_per_side = [int(units) for units in cells_per_side[0][::-1]]

        cells_per_side = tuple(cells_per_side)
    else:
        raise ValueError

    shape = cells_per_side + (-1,) + buffer.shape[1:]

    return torch.reshape(buffer, shape)
