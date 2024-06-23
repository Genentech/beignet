import operator

import torch


def chebyshev_series_vandermonde_1d(input, degree):
    ideg = operator.index(degree)

    if ideg < 0:
        raise ValueError

    input = torch.ravel(input) + 0.0

    output = torch.empty((ideg + 1,) + input.shape, dtype=input.dtype)

    output[0] = input * 0 + 1

    if ideg > 0:
        output[1] = input

        for index in range(2, ideg + 1):
            output[index] = output[index - 1] * 2 * input - output[index - 2]

    output = torch.moveaxis(output, 0, -1)

    return output
