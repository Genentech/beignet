import torch

from .__as_series import _as_series
from ._add_probabilists_hermite_series import add_probabilists_hermite_series
from ._multiply_probabilists_hermite_series_by_x import (
    multiply_probabilists_hermite_series_by_x,
)
from ._subtract_probabilists_hermite_series import subtract_probabilists_hermite_series


def multiply_probabilists_hermite_series(input, other):
    (
        input,
        other,
    ) = _as_series([input, other])

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
            c0 = subtract_probabilists_hermite_series(c[-i] * xs, input * (nd - 1))
            input = add_probabilists_hermite_series(
                tmp, multiply_probabilists_hermite_series_by_x(input)
            )

    output = multiply_probabilists_hermite_series_by_x(input)

    output = add_probabilists_hermite_series(c0, output)

    return output
