from typing import Literal

import torch
from torch import Tensor

from .__as_series import _as_series
from ._lagadd import lagadd
from ._lagmulx import lagmulx
from ._lagsub import lagsub


def lagmul(
    input: Tensor,
    other: Tensor,
    mode: Literal["full", "same", "valid"] = "full",
) -> Tensor:
    [input, other] = _as_series([input, other])

    m, n = input.shape[0], other.shape[0]

    if m > n:
        x, y = other, input
    else:
        x, y = input, other

    match x.shape[0]:
        case 1:
            a = lagadd(torch.zeros(m + n - 1), x[0] * y)
            b = torch.zeros(m + n - 1)
        case 2:
            a = lagadd(torch.zeros(m + n - 1), x[0] * y)
            b = lagadd(torch.zeros(m + n - 1), x[1] * y)
        case _:
            size = x.shape[0]

            a = lagadd(torch.zeros(m + n - 1), x[-2] * y)
            b = lagadd(torch.zeros(m + n - 1), x[-1] * y)

            for i in range(3, x.shape[0] + 1):
                previous = a

                size = size - 1

                a = lagsub(x[-i] * y, (b * (size - 1.0)) / size)
                b = lagadd(
                    previous, lagsub((2.0 * size - 1.0) * b, lagmulx(b, "same")) / size
                )

    output = lagadd(a, lagsub(b, lagmulx(b, "same")))

    if mode == "same":
        output = output[: max(m, n)]

    return output
