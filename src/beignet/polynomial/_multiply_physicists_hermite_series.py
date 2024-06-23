import torch

from .__as_series import _as_series
from ._add_physicists_hermite_series import add_physicists_hermite_series
from ._multiply_physicists_hermite_series_by_x import (
    multiply_physicists_hermite_series_by_x,
)
from ._subtract_physicists_hermite_series import subtract_physicists_hermite_series


def multiply_physicists_hermite_series(input, other):
    input, other = _as_series([input, other])

    if len(input) > len(other):
        c = other
        xs = input
    else:
        c = input
        xs = other

    if len(c) == 1:
        c0 = c[0] * xs
        input = 0
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
            c0 = subtract_physicists_hermite_series(c[-i] * xs, input * (2 * (nd - 1)))
            input = add_physicists_hermite_series(
                tmp, multiply_physicists_hermite_series_by_x(input) * 2
            )

    input = torch.tensor(input)
    input = torch.ravel(input)

    output = multiply_physicists_hermite_series_by_x(input) * 2

    output = add_physicists_hermite_series(c0, output)

    return output
