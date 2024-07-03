import numpy
import torch
from torch import Tensor

from ._evaluate_chebyshev_polynomial import evaluate_chebyshev_polynomial


def integrate_chebyshev_polynomial(
    input: Tensor,
    order=1,
    k=None,
    lower_bound=0,
    scale=1,
    axis=0,
) -> Tensor:
    if k is None:
        k = []

    input = torch.atleast_1d(input)

    lower_bound = torch.tensor(lower_bound)

    scale = torch.tensor(scale)

    if not numpy.iterable(k):
        k = [k]

    if len(k) > order:
        raise ValueError

    if lower_bound.ndim != 0:
        raise ValueError

    if scale.ndim != 0:
        raise ValueError

    if order == 0:
        return input

    input = torch.moveaxis(input, axis, 0)

    k = torch.tensor([*k] + [0.0] * (order - len(k)))

    k = torch.atleast_1d(k)

    for i in range(order):
        n = input.shape[0]

        input = input * scale

        tmp = torch.empty([n + 1, *input.shape[1:]])

        tmp[0] = input[0] * 0
        tmp[1] = input[0]

        if n > 1:
            tmp[2] = input[1] / 4

        if n < 2:
            j = torch.tensor([], dtype=torch.int32)
        else:
            j = torch.arange(2, n)

        tmp[j + 1] = (input[j].T / (2 * (j + 1))).T
        tmp[j - 1] = tmp[j - 1] + -(input[j] / (2 * (j - 1)))

        tmp[0] = tmp[0] + (k[i] - evaluate_chebyshev_polynomial(lower_bound, tmp))

        input = tmp

    input = torch.moveaxis(input, 0, axis)

    return input
