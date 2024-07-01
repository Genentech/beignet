import numpy
import torch
from torch import Tensor

from beignet.polynomial import _as_series, polyval
from beignet.polynomial.__pad_along_axis import _pad_along_axis


def polyint(
    input: Tensor,
    order: int = 1,
    k=None,
    lower_bound: float = 0,
    scale: float = 1,
    dim: int = 0,
) -> Tensor:
    r"""
    Parameters
    ----------
    input : Tensor
        Polynomial coefficients.

    order : int, default=1

    k : int, optional

    lower_bound : float, default=0

    scale : float, default=1

    dim : int, default=0

    Returns
    -------
    output : Tensor
        Polynomial coefficients of the integral.
    """
    if k is None:
        k = []

    [input] = _as_series([input])

    lower_bound, scale = map(torch.tensor, (lower_bound, scale))

    if not numpy.iterable(k):
        k = [k]

    if order < 0:
        raise ValueError

    if len(k) > order:
        raise ValueError

    if lower_bound.ndim != 0:
        raise ValueError

    if scale.ndim != 0:
        raise ValueError

    if order == 0:
        return input

    k = torch.tensor(list(k) + [0] * (order - len(k)))
    k = torch.atleast_1d(k)

    n = input.shape[dim]

    input = _pad_along_axis(
        input,
        torch.tensor([0, order]),
        dim,
    )

    input = torch.moveaxis(input, dim, 0)

    d = torch.arange(n + order) + 1

    for i in range(0, order):
        input = input * scale

        input = (input.T / d).T

        input = torch.roll(input, 1, dims=[0])

        input[0] = 0.0

        input[0] += k[i] - polyval(lower_bound, input)

    return torch.moveaxis(input, 0, dim)
