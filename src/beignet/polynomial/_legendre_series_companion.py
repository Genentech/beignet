import torch

from .__as_series import _as_series


def legendre_series_companion(c):
    [c] = _as_series([c])
    if len(c) < 2:
        raise ValueError
    if len(c) == 2:
        return torch.tensor([[-c[0] / c[1]]])

    n = len(c) - 1
    output = torch.zeros((n, n), dtype=c.dtype)
    scl = 1.0 / torch.sqrt(2 * torch.arange(n) + 1)
    top = output.reshape(-1)[1 :: n + 1]
    bot = output.reshape(-1)[n :: n + 1]
    top[...] = torch.arange(1, n) * scl[: n - 1] * scl[1:n]
    bot[...] = top

    output[:, -1] -= (c[:-1] / c[-1]) * (scl / scl[-1]) * (n / (2 * n - 1))

    return output
