import torch
from torch import Tensor

from .__as_series import _as_series
from ._add_power_series import add_power_series
from ._multiply_power_series_by_x import multiply_power_series_by_x
from ._subtract_power_series import subtract_power_series


def physicists_hermite_series_to_power_series(input: Tensor) -> Tensor:
    (input,) = _as_series([input])

    n = len(input)

    if n == 1:
        return input

    if n == 2:
        input[1] *= 2

        return input

    a = input[-2]
    b = input[-1]

    a = torch.ravel(a)
    b = torch.ravel(b)

    for index in range(n - 1, 1, -1):
        c = a

        aa = input[index - 2]
        ab = b * (2 * (index - 1))

        aa = torch.ravel(aa)
        ab = torch.ravel(ab)

        a = subtract_power_series(aa, ab)

        b = multiply_power_series_by_x(b)
        b = b * 2
        b = add_power_series(c, b)

    output = multiply_power_series_by_x(b) * 2

    output = add_power_series(a, output)

    return output
