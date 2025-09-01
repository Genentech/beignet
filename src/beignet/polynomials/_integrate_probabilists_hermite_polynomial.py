import numpy
import torch

from beignet.polynomials._evaluate_probabilists_hermite_polynomial import (
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

    lower_bound = torch.as_tensor(lower_bound)
    scale = torch.as_tensor(scale)

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

        # Broadcast j along the leading dimension without using deprecated .T
        denom = (j + 1).to(dtype=input.dtype, device=input.device)
        denom = denom.reshape(-1, *([1] * (input[j].ndim - 1)))
        tmp[j + 1] = input[j] / denom

        hermeval_value = torch.as_tensor(
            evaluate_probabilists_hermite_polynomial(lower_bound, tmp)
        )
        tmp[0] += k[i] - hermeval_value

        input = tmp

    return torch.moveaxis(input, 0, axis)
