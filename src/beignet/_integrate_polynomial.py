import numpy
import torch
import torch._numpy._funcs_impl
from torch import Tensor

from ._evaluate_polynomial import evaluate_polynomial


def integrate_polynomial(
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

    input = torch.atleast_1d(input)

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

    padding = torch.tensor([0, order])
    input = torch.moveaxis(input, dim, 0)
    if padding[0] < 0:
        input = input[torch.abs(padding[0]) :]

        padding = (0, padding[1])
    if padding[1] < 0:
        input = input[: -torch.abs(padding[1])]

        padding = (padding[0], 0)
    npad = torch.tensor([(0, 0)] * input.ndim)
    npad[0] = padding
    output = torch._numpy._funcs_impl.pad(
        input,
        pad_width=npad,
        mode="constant",
        constant_values=0,
    )
    input = torch.moveaxis(output, 0, dim)

    input = torch.moveaxis(input, dim, 0)

    d = torch.arange(n + order) + 1

    for i in range(0, order):
        input = input * scale

        input = (input.T / d).T

        input = torch.roll(input, 1, dims=[0])

        input[0] = 0.0

        input[0] += k[i] - evaluate_polynomial(lower_bound, input)

    return torch.moveaxis(input, 0, dim)
