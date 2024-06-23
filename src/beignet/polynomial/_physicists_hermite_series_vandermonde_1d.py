import operator

import torch


def physicists_hermite_series_vandermonde_1d(x, deg):
    ideg = operator.index(deg)
    if ideg < 0:
        raise ValueError

    x = torch.ravel(x) + 0.0
    dims = (ideg + 1,) + x.shape
    dtyp = x.dtype
    v = torch.empty(dims, dtype=dtyp)
    v[0] = x * 0 + 1
    if ideg > 0:
        x2 = x * 2
        v[1] = x2
        for i in range(2, ideg + 1):
            v[i] = v[i - 1] * x2 - v[i - 2] * (2 * (i - 1))
    return torch.moveaxis(v, 0, -1)
