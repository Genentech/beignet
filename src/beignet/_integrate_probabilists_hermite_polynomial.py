import numpy
import torch

from ._evaluate_probabilists_hermite_polynomial import (
    evaluate_probabilists_hermite_polynomial,
)


def integrate_probabilists_hermite_polynomial(
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
    k = torch.tensor(list(k) + [0] * (order - len(k)))
    k = torch.atleast_1d(k)

    for i in range(order):
        n = input.shape[0]
        input *= scale
        tmp = torch.empty((n + 1,) + input.shape[1:], dtype=input.dtype)

        tmp[0] = input[0] * 0
        tmp[1] = input[0]

        j = torch.arange(1, n)

        tmp[j + 1] = (input[j].T / (j + 1)).T

        hermeval_value = torch.tensor(
            evaluate_probabilists_hermite_polynomial(lower_bound, tmp)
        )
        tmp[0] += k[i] - hermeval_value

        input = tmp

    return torch.moveaxis(input, 0, axis)
