import torch

from beignet.polynomial import _as_series


def legmulx(c):
    [c] = _as_series([c])

    if len(c) == 1 and c[0] == 0:
        return c

    prd = torch.empty(len(c) + 1, dtype=c.dtype)

    prd[0] = c[0] * 0

    prd[1] = c[0]

    for i in range(1, len(c)):
        j = i + 1
        k = i - 1
        s = i + j
        prd[j] = (c[i] * j) / s
        prd[k] += (c[i] * i) / s

    return prd
