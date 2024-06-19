import torch

from .__as_series import _as_series


def lagmulx(c):
    [c] = _as_series([c])

    if len(c) == 1 and c[0] == 0:
        return c

    prd = torch.empty(len(c) + 1, dtype=c.dtype)
    prd[0] = c[0]
    prd[1] = -c[0]
    for i in range(1, len(c)):
        prd[i + 1] = -c[i] * (i + 1)
        prd[i] += c[i] * (2 * i + 1)
        prd[i - 1] -= c[i] * i
    return prd
