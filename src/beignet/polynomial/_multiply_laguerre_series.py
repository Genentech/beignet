import torch
from torch import Tensor

from .__as_series import _as_series
from ._add_laguerre_series import add_laguerre_series
from ._multiply_laguerre_series_by_x import multiply_laguerre_series_by_x
from ._subtract_laguerre_series import subtract_laguerre_series


def multiply_laguerre_series(input: Tensor, other: Tensor) -> Tensor:
    input, other = _as_series([input, other])

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

            c0 = subtract_laguerre_series(c[-i] * xs, (input * (nd - 1)) / nd)

            nd_input = (2 * nd - 1) * input

            t = multiply_laguerre_series_by_x(input)

            input_nd = subtract_laguerre_series(nd_input, t)

            input_nd = input_nd / nd

            input = add_laguerre_series(tmp, input_nd)

    output = multiply_laguerre_series_by_x(input)

    output = subtract_laguerre_series(input, output)

    output = add_laguerre_series(c0, output)

    return output
