import torch

from .__as_series import _as_series


def chebmulx(c):
    [c] = _as_series([c])

    if len(c) == 1 and c[0] == 0:
        return c

    output = torch.empty(len(c) + 1, dtype=c.dtype)

    output[0] = c[0] * 0
    output[1] = c[0]

    if len(c) > 1:
        tmp = c[1:] / 2

        output[2:] = tmp
        output[0:-2] += tmp

    return output
