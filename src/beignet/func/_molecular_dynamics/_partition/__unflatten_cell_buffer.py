import torch

def _unflatten_cell_buffer(buffer: torch.Tensor, cells_per_side: [int, float, torch.Tensor], dim: int) -> torch.Tensor:
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

    # Check and standardize cells_per_side to a tuple
    if isinstance(cells_per_side, (int, float)):
        cells_per_side = (int(cells_per_side),) * dim
    elif isinstance(cells_per_side, torch.Tensor):
        cells_per_side = cells_per_side.to(dtype=torch.int)
        if cells_per_side.ndimension() == 0:
            cells_per_side = (cells_per_side.item(),) * dim
        elif cells_per_side.ndimension() == 1:
            cells_per_side = tuple(cells_per_side.flip(0).tolist())
        elif cells_per_side.ndimension() == 2:
            cells_per_side = tuple(cells_per_side[0].flip(0).tolist())
        else:
            raise ValueError("cells_per_side must be a scalar, 1D tensor, or 2D tensor.")
    else:
        raise ValueError("Unsupported type for cells_per_side.")

    # Calculate product of dimensions in cells_per_side
    total_cells = 1
    for size in cells_per_side:
        total_cells *= size

    # Ensure the buffer can be reshaped into the target shape
    if buffer.numel() % total_cells != 0:
        raise ValueError("Buffer size is not compatible with the desired shape.")

    inner_dim_size = buffer.numel() // total_cells

    # Reshape the buffer
    new_shape = cells_per_side + (inner_dim_size,) + buffer.shape[1:]
    return buffer.view(*new_shape)