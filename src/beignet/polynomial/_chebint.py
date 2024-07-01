import numpy
import torch
from torch import Tensor

from beignet.polynomial import _as_series, chebval


def chebint(
    c: Tensor,
    order=1,
    k=None,
    lower_bound=0,
    scale=1,
    axis=0,
) -> Tensor:
    if k is None:
        k = []

    [c] = _as_series([c])

    lower_bound = torch.tensor(lower_bound)

    scale = torch.tensor(scale)

    if not numpy.iterable(k):
        k = [k]

    if len(k) > order:
        raise ValueError

    if lower_bound.ndim != 0:
        raise ValueError

    if scale.ndim != 0:
        raise ValueError

    if order == 0:
        return c

    c = torch.moveaxis(c, axis, 0)

    k = torch.tensor([*k] + [0.0] * (order - len(k)))

    k = torch.atleast_1d(k)

    for i in range(order):
        n = c.shape[0]

        c = c * scale

        tmp = torch.empty([n + 1, *c.shape[1:]])

        tmp[0] = c[0] * 0
        tmp[1] = c[0]

        if n > 1:
            tmp[2] = c[1] / 4

        if n < 2:
            j = torch.tensor([], dtype=torch.int32)
        else:
            j = torch.arange(2, n)

        tmp[j + 1] = (c[j].T / (2 * (j + 1))).T
        tmp[j - 1] = tmp[j - 1] + -(c[j] / (2 * (j - 1)))

        tmp[0] = tmp[0] + (k[i] - chebval(lower_bound, tmp))

        c = tmp

    c = torch.moveaxis(c, 0, axis)

    return c
