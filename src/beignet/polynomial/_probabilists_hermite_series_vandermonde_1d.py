import operator

import torch


def probabilists_hermite_series_vandermonde_1d(x, deg):
    ideg = operator.index(deg)

    if ideg < 0:
        raise ValueError

    x = torch.ravel(x) + 0.0

    output = torch.empty((ideg + 1,) + x.shape, dtype=x.dtype)

    output[0] = x * 0 + 1

    if ideg > 0:
        output[1] = x

        for index in range(2, ideg + 1):
            output[index] = output[index - 1] * x - output[index - 2] * (index - 1)

    output = torch.moveaxis(output, 0, -1)

    return output
