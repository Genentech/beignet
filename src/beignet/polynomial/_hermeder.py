import torch
from torch import Tensor

from beignet.polynomial import _as_series


def hermeder(
    c,
    order=1,
    scale=1,
    axis=0,
) -> Tensor:
    if order < 0:
        raise ValueError

    [c] = _as_series([c])

    if order == 0:
        return c

    c = torch.moveaxis(c, axis, 0)

    n = c.shape[0]

    if order >= n:
        c = torch.zeros_like(c[:1])
    else:
        for _ in range(order):
            n = n - 1

            c = c * scale

            der = torch.empty((n,) + c.shape[1:], dtype=c.dtype)

            j = torch.arange(n, 0, -1)

            der[j - 1] = (j * (c[j]).T).T

            c = der

    c = torch.moveaxis(c, 0, axis)

    return c
