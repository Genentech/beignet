import torch
from torch import Tensor

from src.beignet.func._molecular_dynamics._partition.__safe_mask import \
  safe_mask
from src.beignet.func._molecular_dynamics._partition.__square_distance import \
  square_distance


def distance(dR: Tensor) -> Tensor:
    """Computes distances.

    Args:
      dR: Matrix of displacements; `Tensor(shape=[..., spatial_dim])`.
    Returns:
      Matrix of distances; `Tensor(shape=[...])`.
    """
    dr = square_distance(dR)

    return safe_mask(dr > 0, torch.sqrt, dr)