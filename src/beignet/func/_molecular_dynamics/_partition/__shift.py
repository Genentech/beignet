import torch
from torch import Tensor


def _shift(a: Tensor, b: Tensor) -> Tensor:
    """
    Shifts a tensor `a` along dimensions specified in `b`.

    The shift can be applied in up to three dimensions (x, y, z).
    Positive values in `b` indicate a forward shift, while negative values indicate
    a backward shift.

    Parameters
    ----------
    a : Tensor
      The input tensor to be shifted.
    b : Tensor
      A tensor of two or three elements specifying the shift amount for each dimension.

    Returns
    -------
    Tensor
      The shifted tensor.
    """

    return torch.roll(a, shifts=tuple(b), dims=tuple(range(len(b))))
