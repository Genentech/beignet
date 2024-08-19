import torch
from torch import Tensor


def iota(shape: tuple[int, ...], dim: int = 0, **kwargs) -> Tensor:
    r"""Generate a tensor with a specified shape where elements along the given dimension
    are sequential integers starting from 0.

    Parameters
    ----------
    shape : tuple[int, ...]
        The shape of the resulting tensor.
    dim : int, optional
        The dimension along which to vary the values (default is 0).

    Returns
    -------
    Tensor
        A tensor of the specified shape with sequential integers along the specified dimension.

    Raises
    ------
    IndexError
        If `dim` is out of the range of `shape`.
    """
    dimensions = []

    for index, _ in enumerate(shape):
        if index != dim:
            dimension = 1

        else:
            dimension = shape[index]

        dimensions = [*dimensions, dimension]

    return torch.arange(shape[dim], **kwargs).view(*dimensions).expand(*shape)
