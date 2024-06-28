import torch
from torch import Tensor


def _hash_constants(spatial_dimensions: int, cells_per_side: Tensor) -> Tensor:
    r"""Compute constants used for hashing in a spatial partitioning scheme.

    The function calculates constants that help in determining the hash value
    for a given cell in an N-dimensional grid, based on the number of cells
    per side in each dimension.

    Parameters
    ----------
    spatial_dimensions : int
        The number of spatial dimensions.
    cells_per_side : Tensor
        A 1D tensor indicating the number of cells per side in each dimension.
        If the size of the tensor is 0, it is assumed that the grid is uniform
        in all dimensions, and this single value will be used for all dimensions.
        If the number of elements matches `spatial_dimensions`, these values
        specify the number of cells per side for each corresponding dimension.

    Returns
    -------
    Tensor
        A 2D tensor of shape (1, spatial_dimensions) containing the computed hash
        constants, used to index cells in the grid.

    Raises
    ------
    ValueError
        If the size of `cells_per_side` is not zero or `spatial_dimensions`.
    """
    if cells_per_side.numel() == 1:
        constants = [[cells_per_side**dim for dim in range(spatial_dimensions)]]
        return torch.tensor(constants, dtype=torch.int32)

    elif cells_per_side.numel() == spatial_dimensions:
        one = torch.tensor([[1]], dtype=torch.int32)
        cells_per_side = torch.cat((one, cells_per_side[:, :-1]), dim=1)
        return torch.cumprod(cells_per_side, dim=1).squeeze()

    else:
        raise ValueError(
            "Cells per side must either: have 0 dimensions, or be the same size as spatial dimensions."
        )
