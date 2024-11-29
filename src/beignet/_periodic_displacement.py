import torch
from torch import Tensor


def periodic_displacement(box: float | Tensor, dR: Tensor) -> Tensor:
    r"""Wraps displacement vectors into a hypercube.

    Parameters
    ----------
    box : float or Tensor
        Specification of hypercube size. Either:
        (a) float if all sides have equal length.
        (b) Tensor of shape (spatial_dim,) if sides have different lengths.
    dR : Tensor
        Matrix of displacements with shape (..., spatial_dim).

    Returns
    -------
    Tensor
        Matrix of wrapped displacements with shape (..., spatial_dim).
    """
    distances = (
        torch.remainder(dR + box * 0.5, box)
        - 0.5 * box
    )
    return distances
