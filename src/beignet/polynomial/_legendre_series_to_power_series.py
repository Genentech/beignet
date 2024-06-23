import torch
from torch import Tensor

from .__as_series import _as_series
from ._add_power_series import add_power_series
from ._multiply_power_series_by_x import multiply_power_series_by_x
from ._subtract_power_series import subtract_power_series


def legendre_series_to_power_series(input: Tensor) -> Tensor:
    (input,) = _as_series([input])

    n = len(input)

    if n < 3:
        return input

    a = torch.ravel(input[-2])
    b = torch.ravel(input[-1])

    for index in range(n - 1, 1, -1):
        c = a
        e = torch.ravel(input[index - 2])
        f = (b * (index - 1)) / index
        a = subtract_power_series(e, f)
        g = multiply_power_series_by_x(b) * (2 * index - 1)
        d = g / index
        b = add_power_series(c, d)

    x = multiply_power_series_by_x(b)

    output = add_power_series(a, x)

    return output
