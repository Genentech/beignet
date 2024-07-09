import numpy
import torch

from ._evaluate_laguerre_polynomial import evaluate_laguerre_polynomial


def integrate_laguerre_polynomial(
    input,
    order=1,
    k=None,
    lower_bound=0,
    scale=1,
    axis=0,
):
    if k is None:
        k = []

    input = torch.atleast_1d(input)

    lower_bound, scale = map(torch.tensor, (lower_bound, scale))

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
    k = torch.tensor(list(k) + [0] * (order - len(k)))
    k = torch.atleast_1d(k)

    for i in range(order):
        n = input.shape[0]
        input *= scale

        tmp = torch.empty((n + 1,) + input.shape[1:], dtype=input.dtype)

        tmp[0] = input[0]
        tmp[1] = -input[0]

        j = torch.arange(1, n)

        tmp[j] += input[j]
        tmp[j + 1] += -input[j]

        tmp_value = torch.tensor(evaluate_laguerre_polynomial(lower_bound, tmp))
        tmp[0] += k[i] - tmp_value

        input = tmp

    input = torch.moveaxis(input, 0, axis)
    return input
