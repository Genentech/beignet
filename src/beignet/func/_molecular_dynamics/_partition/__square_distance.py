import torch
from torch import Tensor


def square_distance(input: Tensor) -> Tensor:
    """Computes square distances.

    Args:
    input: Matrix of displacements; `Tensor(shape=[..., spatial_dim])`.
    Returns:
    Matrix of squared distances; `Tensor(shape=[...])`.
    """
    return torch.sum(input**2, dim=-1)
