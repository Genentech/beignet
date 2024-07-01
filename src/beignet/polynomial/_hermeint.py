import numpy
import torch

from beignet.polynomial import _as_series, hermeval


def hermeint(
    c,
    order=1,
    k=None,
    lower_bound=0,
    scale=1,
    axis=0,
):
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
    k = torch.tensor(list(k) + [0] * (order - len(k)))
    k = torch.atleast_1d(k)

    for i in range(order):
        n = c.shape[0]
        c *= scale
        tmp = torch.empty((n + 1,) + c.shape[1:], dtype=c.dtype)

        tmp[0] = c[0] * 0
        tmp[1] = c[0]

        j = torch.arange(1, n)

        tmp[j + 1] = (c[j].T / (j + 1)).T

        hermeval_value = torch.tensor(hermeval(lower_bound, tmp))
        tmp[0] += k[i] - hermeval_value

        c = tmp

    return torch.moveaxis(c, 0, axis)
