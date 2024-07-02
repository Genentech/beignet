import torch
from torch import Tensor

from .__as_series import _as_series
from ._polyadd import polyadd
from ._polymulx import polymulx
from ._polysub import polysub


def lag2poly(
    c: Tensor,
) -> Tensor:
    [c] = _as_series([c])

    n = c.shape[0]

    if n == 1:
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

            c0 = polysub(c[i - 2], (c1 * (i - 1)) / i)

            c1 = polyadd(tmp, polysub((2 * i - 1) * c1, polymulx(c1, "same")) / i)

            return c0, c1

        b = n - 2

        x = (c0, c1)

        y = x

        for index in range(0, b):
            y = body(index, y)

        c0, c1 = y

        return polyadd(c0, polysub(c1, polymulx(c1, "same")))
