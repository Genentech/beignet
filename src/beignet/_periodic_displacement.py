import torch
from torch import Tensor


def periodic_displacement(input: Tensor, position: Tensor) -> Tensor:
    r"""Wraps displacement vectors into a hypercube.

    Parameters:
    ----------
    box : float or Tensor
        Specification of hypercube size. Either:
        (a) scalar if all sides have equal length.
        (b) Tensor of shape (spatial_dim,) if sides have different lengths.

    dR : Tensor
        Matrix of displacements with shape (..., spatial_dim).

    Returns:
    -------
    output : Tensor, shape=(...)
        Matrix of wrapped displacements with shape (..., spatial_dim).
    """
    output = torch.remainder(position + input * 0.5, input) - 0.5 * input
    return output
