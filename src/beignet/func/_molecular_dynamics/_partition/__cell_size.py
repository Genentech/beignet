import torch
from torch import Tensor


def _cell_size(box: Tensor, minimum_unit_size: Tensor) -> Tensor:
    r"""Compute the size of cells within a box based on the minimum unit size.

    Parameters:
    -----------
    box : Tensor
        The size of the box. This must be a Tensor.
    minimum_unit_size : Tensor
        The minimum size of the units (cells). This must be a Tensor of the same shape as `box` or a scalar Tensor.

    Returns:
    --------
    Tensor
        The size of the cells in the box.

    Raises:
    -------
    ValueError
        If the box and minimum unit size do not have the same shape and `minimum_unit_size` is not a scalar.
    """
    if box.shape == minimum_unit_size.shape or minimum_unit_size.ndim == 0:
        return box / torch.floor(box / minimum_unit_size)

    else:
        raise ValueError("Box and minimum unit size must be of the same shape.")
