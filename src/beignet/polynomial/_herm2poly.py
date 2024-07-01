import torch
from torch import Tensor

from beignet.polynomial import _as_series, polyadd, polymulx, polysub


def herm2poly(
    c: Tensor,
) -> Tensor:
    [c] = _as_series([c])
    n = c.shape[0]

    if n == 1:
        return c

    if n == 2:
        c[1] = c[1] * 2

        return c
    else:
        c0 = torch.zeros_like(c)
        c0[0] = c[-2]

        c1 = torch.zeros_like(c)
        c1[0] = c[-1]

        def body(k, c0c1):
            i = n - 1 - k
            c0, c1 = c0c1
            tmp = c0
            c0 = polysub(c[i - 2], c1 * (2 * (i - 1)))
            c1 = polyadd(tmp, polymulx(c1, "same") * 2)
            return c0, c1

        x = (c0, c1)

        y = x

        for index in range(0, n - 2):
            y = body(index, y)

        c0, c1 = y

        return polyadd(c0, polymulx(c1, "same") * 2)
