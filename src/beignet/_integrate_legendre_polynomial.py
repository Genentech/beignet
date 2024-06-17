import numpy
import torch

from ._evaluate_legendre_polynomial import evaluate_legendre_polynomial


def integrate_legendre_polynomial(
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

    lbnd, scl = map(torch.tensor, (lower_bound, scale))

    if not numpy.iterable(k):
        k = [k]

    if len(k) > order:
        raise ValueError("Too many integration constants")

    if lbnd.ndim != 0:
        raise ValueError("lbnd must be a scalar.")

    if scl.ndim != 0:
        raise ValueError("scl must be a scalar.")

    if order == 0:
        return input

    output = torch.moveaxis(input, axis, 0)

    k = torch.tensor(list(k) + [0] * (order - len(k)))
    k = torch.atleast_1d(k)

    for i in range(order):
        n = len(output)
        output *= scl
        tmp = torch.empty((n + 1,) + output.shape[1:], dtype=output.dtype)
        tmp[0] = output[0] * 0
        tmp[1] = output[0]
        if n > 1:
            tmp[2] = output[1] / 3

        if n < 2:
            j = torch.tensor([], dtype=torch.int32)
        else:
            j = torch.arange(2, n)

        t = (output[j].T / (2 * j + 1)).T
        tmp[j + 1] = t
        tmp[j - 1] += -t
        legval_value = evaluate_legendre_polynomial(lbnd, tmp)
        tmp[0] += k[i] - legval_value
        output = tmp

    output = torch.moveaxis(output, 0, axis)

    return output
