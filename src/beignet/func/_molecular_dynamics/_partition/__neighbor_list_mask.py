import torch
from torch import Tensor

from src.beignet.func._molecular_dynamics._partition.__is_neighbor_list_sparse import \
    is_neighbor_list_sparse
from src.beignet.func._molecular_dynamics._partition.__neighbor_list import \
    _NeighborList


def neighbor_list_mask(neighbor: _NeighborList, mask_self: bool = False) -> Tensor:
  r"""Compute a mask for neighbor list.

  Parameters
    ----------
    neighbor : _NeighborList
      The input tensor to be shifted.
    mask_self : bool
      A tensor of two or three elements specifying the shift amount for each dimension.

    Returns
    -------
    mask : Tensor
      The shifted tensor.
  """
  if is_neighbor_list_sparse(neighbor.format):
    mask = neighbor.indexes[0] < len(neighbor.reference_positions)

    if mask_self:
      mask = mask & (neighbor.indexes[0] != neighbor.indexes[1])

    return mask

  mask = neighbor.indexes < len(neighbor.indexes)

  if mask_self:
    N = len(neighbor.reference_positions)

    self_mask = neighbor.indexes != torch.arange(N).view(N, 1)

    mask = mask & self_mask

  return mask