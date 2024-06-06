import torch
from torch import Tensor


def _unflatten_cell_buffer(buffer: Tensor, cells_per_side: [int, float, Tensor], dim: int):
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
    # if cells per side is an int, float, or a tensor with 0 dimensions (scalar) -> return tuple
    if (isinstance(cells_per_side, int) or isinstance(cells_per_side, float) or (torch.is_tensor(cells_per_side) and not cells_per_side.shape)):
        cells_per_side = (int(cells_per_side),) * dim

    # if cells per side is a 1D tensor, reverse the tensor, and place each elem into a tuple
    elif torch.is_tensor(cells_per_side) and len(cells_per_side.shape) == 1:
        reversed_cells_per_side = torch.flip(cells_per_side, [0])

        cells_per_side = tuple(int(x) for x in reversed_cells_per_side)

    elif torch.is_tensor(cells_per_side) and len(cells_per_side.shape) == 2:
        reversed_first_row = torch.flip(cells_per_side[0], [0])

        cells_per_side = tuple(int(x) for x in reversed_first_row)

    else:
        raise ValueError()

    # creates a tuple (cells_per_side[0], ..., cells_per_side[n], -1, buffer.shape[1], ..., buffer.shape[m])
    new_shape = cells_per_side + (-1,) + buffer.shape[1:]

    # Calculate the total number of elements
    total_elements = buffer.numel()

    # Calculate the inferred dimension manually
    known_dims_product = 1
    for dim in new_shape:
        if dim != -1:
            known_dims_product *= dim

    # Infer the -1 dimension
    inferred_dim = total_elements // known_dims_product

    if total_elements % known_dims_product != 0:
        raise ValueError(
            f"Inferred dimension {-1} mismatch size! Ensure correct element distribution aligning devices")

    # Replace -1 with the inferred dimension
    new_shape = tuple(inferred_dim if dim == -1 else dim for dim in new_shape)

    # Reshape the buffer with the new shape
    reshaped_buffer = buffer.reshape(new_shape)

    return reshaped_buffer
