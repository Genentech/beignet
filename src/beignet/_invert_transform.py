import torch
from torch import Tensor


def invert_transform(transform: Tensor) -> Tensor:
    """
    Calculates the inverse of an affine transformation matrix.

    Parameters
    ----------
    transform : Tensor
        The affine transformation matrix to be inverted.

    Returns
    -------
    Tensor
        The inverse of the given affine transformation matrix.
    """
    if transform.ndim in {0, 1}:
        return 1.0 / transform

    if transform.ndim == 2:
        return torch.linalg.inv(transform)

    raise ValueError
