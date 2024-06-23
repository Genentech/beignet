import operator

import torch


def chebyshev_series_vandermonde_1d(input, degree):
    ideg = operator.index(degree)

    if ideg < 0:
        raise ValueError

    input = torch.ravel(input) + 0.0

    dims = (ideg + 1,) + input.shape

    v = torch.empty(dims, dtype=input.dtype)

    v[0] = input * 0 + 1

    if ideg > 0:
        x2 = 2 * input
        v[1] = input

        for index in range(2, ideg + 1):
            v[index] = v[index - 1] * x2 - v[index - 2]

    return torch.moveaxis(v, 0, -1)
