import math

import torch

from .__as_series import _as_series


def chebyshev_series_companion(input):
    (input,) = _as_series([input])

    if len(input) < 2:
        raise ValueError

    if len(input) == 2:
        return torch.tensor([[-input[0] / input[1]]])

    n = len(input) - 1

    output = torch.zeros((n, n), dtype=input.dtype)

    scl = torch.tensor([1.0] + [math.sqrt(0.5)] * (n - 1))

    top = torch.reshape(output, [-1])[1 :: n + 1]

    bot = torch.reshape(output, [-1])[n :: n + 1]

    top[0] = math.sqrt(0.5)

    top[1:] = 1 / 2

    bot[...] = top

    output[:, -1] -= (input[:-1] / input[-1]) * (scl / scl[-1]) * 0.5

    return output
