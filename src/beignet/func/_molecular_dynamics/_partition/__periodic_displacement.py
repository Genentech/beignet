import torch
from torch import Tensor


def periodic_displacement(box: float | Tensor, dR: Tensor) -> Tensor:
    """Wraps displacement vectors into a hypercube.

    Args:
    box: Specification of hypercube size. Either,
      (a) float if all sides have equal length.
      (b) Tensor(spatial_dim) if sides have different lengths.
    dR: Matrix of displacements; `Tensor(shape=[..., spatial_dim])`.
    Returns:
    Matrix of wrapped displacements; `ndarray(shape=[..., spatial_dim])`.
    """
    distances = torch.remainder(dR + box * torch.tensor(0.5, dtype=torch.float32), box) - torch.tensor(0.5, dtype=torch.float32) * box
    return distances