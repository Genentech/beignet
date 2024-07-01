import torch
from torch import Tensor

from beignet.polynomial import _as_series


def legder(
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
            c *= scale
            der = torch.empty((n,) + c.shape[1:], dtype=c.dtype)

            def body(k, der_c, n=n):
                j = n - k

                der, c = der_c

                der[j - 1] = (2 * j - 1) * c[j]

                c[j - 2] += c[j]

                return der, c

            b = n - 2

            x = (der, c)

            y = x

            for index in range(0, b):
                y = body(index, y)

            der, c = y

            if n > 1:
                der[1] = 3 * c[2]

            der[0] = c[1]

            c = der

    c = torch.moveaxis(c, 0, axis)

    return c
