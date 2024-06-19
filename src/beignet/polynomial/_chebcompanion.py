import math

import torch

from .__as_series import _as_series


def chebcompanion(c):
    [c] = _as_series([c])
    if len(c) < 2:
        raise ValueError("Series must have maximum degree of at least 1.")
    if len(c) == 2:
        return torch.tensor([[-c[0] / c[1]]])

    n = len(c) - 1
    mat = torch.zeros((n, n), dtype=c.dtype)
    scl = torch.tensor([1.0] + [math.sqrt(0.5)] * (n - 1))
    top = mat.reshape(-1)[1 :: n + 1]
    bot = mat.reshape(-1)[n :: n + 1]
    top[0] = math.sqrt(0.5)
    top[1:] = 1 / 2
    bot[...] = top
    mat[:, -1] -= (c[:-1] / c[-1]) * (scl / scl[-1]) * 0.5
    return mat
