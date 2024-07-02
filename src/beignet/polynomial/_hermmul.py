from typing import Literal

import torch
from torch import Tensor

from .__as_series import _as_series
from ._hermadd import hermadd
from ._hermmulx import hermmulx
from ._hermsub import hermsub


def hermmul(
    input: Tensor, other: Tensor, mode: Literal["full", "same", "valid"] = "full"
) -> Tensor:
    [input, other] = _as_series([input, other])

    m, n = input.shape[0], other.shape[0]

    if m > n:
        x, y = other, input
    else:
        x, y = input, other

    match x.shape[0]:
        case 1:
            a = hermadd(torch.zeros(m + n - 1), x[0] * y)
            b = torch.zeros(m + n - 1)
        case 2:
            a = hermadd(torch.zeros(m + n - 1), x[0] * y)
            b = hermadd(torch.zeros(m + n - 1), x[1] * y)
        case _:
            size = x.shape[0]

            a = hermadd(torch.zeros(m + n - 1), x[-2] * y)
            b = hermadd(torch.zeros(m + n - 1), x[-1] * y)

            for i in range(3, x.shape[0] + 1):
                previous = a

                size = size - 1

                a = hermsub(x[-i] * y, b * (2 * (size - 1.0)))

                b = hermadd(previous, hermmulx(b, "same") * 2.0)

    output = hermadd(a, hermmulx(b, "same") * 2)

    if mode == "same":
        output = output[: max(m, n)]

    return output
