import torch
from torch import Tensor


def periodic_displacement(side: Tensor, dR: Tensor) -> Tensor:
  """Wraps displacement vectors into a hypercube.

  Args:
    side: Specification of hypercube size. Either,
      (a) float if all sides have equal length.
      (b) tensor(spatial_dim) if sides have different lengths.
    dR: Matrix of displacements; `tensor(shape=[..., spatial_dim])`.
  Returns:
    Matrix of wrapped displacements; `tensor(shape=[..., spatial_dim])`.
  """
  return torch.remainder(dR + side * torch.tensor(0.5), side) - torch.tensor(0.5) * side