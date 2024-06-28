import torch
from torch import Tensor
from typing import Any, Callable


def safe_mask(
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
