import operator

import torch
from torch import Tensor


def probabilists_hermite_series_vandermonde_1d(
    input: Tensor,
    degree,
) -> Tensor:
    degree = operator.index(degree)

    if degree < 0:
        raise ValueError

    input = torch.ravel(input) + 0.0

    output = torch.empty((degree + 1,) + input.shape, dtype=input.dtype)

    output[0] = input * 0 + 1

    if degree > 0:
        output[1] = input

        for index in range(2, degree + 1):
            output[index] = output[index - 1] * input - output[index - 2] * (index - 1)

    output = torch.moveaxis(output, 0, -1)

    return output
