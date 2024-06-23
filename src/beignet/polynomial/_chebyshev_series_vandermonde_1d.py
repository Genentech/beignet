import operator

import torch


def chebyshev_series_vandermonde_1d(input, degree):
    ideg = operator.index(degree)

    if ideg < 0:
        raise ValueError

    input = torch.ravel(input)

    input = input + 0.0

    dims = (ideg + 1,) + input.shape

    dtyp = input.dtype

    v = torch.empty(dims, dtype=dtyp)

    v[0] = input * 0 + 1

    if ideg > 0:
        x2 = 2 * input
        v[1] = input

        for i in range(2, ideg + 1):
            v[i] = v[i - 1] * x2 - v[i - 2]

    return torch.moveaxis(v, 0, -1)
