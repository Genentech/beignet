import torch
from torch import Tensor

from .__as_series import _as_series
from ._add_legendre_series import add_legendre_series
from ._multiply_legendre_series_by_x import multiply_legendre_series_by_x
from ._subtract_legendre_series import subtract_legendre_series


def multiply_legendre_series(input: Tensor, other: Tensor) -> Tensor:
    [input, other] = _as_series([input, other])

    if len(input) > len(other):
        c = other
        xs = input
    else:
        c = input
        xs = other

    if len(c) == 1:
        c0 = c[0] * xs
        input = torch.tensor([0])
    elif len(c) == 2:
        c0 = c[0] * xs
        input = c[1] * xs
    else:
        nd = len(c)
        c0 = c[-2] * xs
        input = c[-1] * xs
        for i in range(3, len(c) + 1):
            tmp = c0
            nd = nd - 1
            c0 = subtract_legendre_series(c[-i] * xs, (input * (nd - 1)) / nd)
            input = add_legendre_series(
                tmp, (multiply_legendre_series_by_x(input) * (2 * nd - 1)) / nd
            )

    output = add_legendre_series(c0, multiply_legendre_series_by_x(input))

    return output
