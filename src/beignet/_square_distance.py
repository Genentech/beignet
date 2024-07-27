from typing import Callable, Any

import torch
from torch import Tensor


def _square_distance(input: Tensor) -> Tensor:
    """Computes square distances.

    Args:
    input: Matrix of displacements; `Tensor(shape=[..., spatial_dim])`.
    Returns:
    Matrix of squared distances; `Tensor(shape=[...])`.
    """
    return torch.sum(input**2, dim=-1)


def _safe_mask(
    mask: Tensor, fn: Callable, operand: Tensor, placeholder: Any = 0
) -> Tensor:
    r"""Applies a function to elements of a tensor where a mask is True, and replaces elements where the mask is False with a placeholder.

    Parameters
    ----------
    mask : Tensor
        A boolean tensor indicating which elements to apply the function to.
    fn : Callable[[Tensor], Tensor]
        The function to apply to the masked elements.
    operand : Tensor
        The tensor to apply the function to.
    placeholder : Any, optional
        The value to use for elements where the mask is False (default is 0).

    Returns
    -------
    Tensor
        A tensor with the function applied to the masked elements and the placeholder value elsewhere.
    """
    masked = torch.where(mask, operand, torch.tensor(0, dtype=operand.dtype))

    return torch.where(mask, fn(masked), torch.tensor(placeholder, dtype=operand.dtype))


def square_distance(dR: Tensor) -> Tensor:
    r"""Computes distances.

    Args:
      dR: Matrix of displacements; `Tensor(shape=[..., spatial_dim])`.
    Returns:
      Matrix of distances; `Tensor(shape=[...])`.
    """
    return _safe_mask(_square_distance(dR) > 0, torch.sqrt, _square_distance(dR))
