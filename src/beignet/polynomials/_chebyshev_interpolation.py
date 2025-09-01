import math
from typing import Callable

import torch
from torch import Tensor


def chebyshev_interpolation(
    func: Callable[[Tensor, ...], Tensor],
    degree: int,
    *args,
):
    if degree < 0:
        raise ValueError

    order = degree + 1

    if order < 1:
        raise ValueError

    zeros = torch.arange(-order + 1, order + 1, 2)

    zeros = torch.sin(zeros * 0.5 * math.pi / order)

    vandermonde = torch.empty([degree + 1, *zeros.shape], dtype=zeros.dtype)

    vandermonde[0] = torch.ones_like(zeros)

    if degree > 0:
        vandermonde[1] = zeros

        for i in range(2, degree + 1):
            v = vandermonde[i - 1] * zeros * 2.0 - vandermonde[i - 2]

            vandermonde[i] = v

    vandermonde = torch.moveaxis(vandermonde, 0, -1)

    output = vandermonde.T @ func(zeros, *args)

    output[0] = output[0] / order

    output[1:] = output[1:] / (order * 0.5)

    return output
