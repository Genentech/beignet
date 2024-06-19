import torch

from .__as_series import _as_series


def chebmulx(c):
    [c] = _as_series([c])

    if len(c) == 1 and c[0] == 0:
        return c

    prd = torch.empty(len(c) + 1, dtype=c.dtype)
    prd[0] = c[0] * 0
    prd[1] = c[0]
    if len(c) > 1:
        tmp = c[1:] / 2
        prd[2:] = tmp
        prd[0:-2] += tmp
    return prd
