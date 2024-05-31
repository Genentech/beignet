import torch
from torch import Tensor


def _unflatten_cell_buffer(
    buffer: Tensor,
    cells_per_side: Tensor,
    dim: int,
) -> Tensor:
    """
    Reshape a flat buffer into a multidimensional cell buffer.

    Parameters
    ----------
    buffer : Tensor
        The input flat buffer tensor to be reshaped.
    cells_per_side : Tensor or int or float
        The number of cells per side in each dimension.
    dim : int
        The number of spatial dimensions.

    Returns
    -------
    Tensor
        The reshaped buffer tensor.
    """
    if (
        isinstance(cells_per_side, int)
        or isinstance(cells_per_side, float)
        or (isinstance(cells_per_side, Tensor) and cells_per_side.ndim == 0)
    ):
        cells_per_side = (int(cells_per_side),) * dim
    elif isinstance(cells_per_side, Tensor) and len(cells_per_side.shape) == 1:
        cells_per_side = [int(units) for units in cells_per_side.flip(0)]

        cells_per_side = tuple(cells_per_side)

    elif isinstance(cells_per_side, Tensor) and len(cells_per_side.shape) == 2:
        cells_per_side = [int(x) for x in cells_per_side[0].flip(0)]

        cells_per_side = tuple(cells_per_side)

    else:
        raise ValueError("Invalid cells_per_side format")

    shape = cells_per_side + (-1,) + buffer.shape[1:]

    return torch.reshape(buffer, shape)
