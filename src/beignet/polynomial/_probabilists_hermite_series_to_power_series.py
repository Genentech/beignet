import torch

from .__as_series import _as_series
from ._add_power_series import add_power_series
from ._multiply_power_series_by_x import multiply_power_series_by_x
from ._subtract_power_series import subtract_power_series


def probabilists_hermite_series_to_power_series(input):
    (input,) = _as_series([input])

    n = len(input)

    if n in {0, 1}:
        return input

    a = input[-2]
    b = input[-1]

    a = torch.ravel(a)
    b = torch.ravel(b)

    for index in range(n - 1, 1, -1):
        c = a

        aa = input[index - 2]
        ab = b * (index - 1)

        aa = torch.ravel(aa)
        ab = torch.ravel(ab)

        a = subtract_power_series(aa, ab)

        b = multiply_power_series_by_x(b)
        b = add_power_series(c, b)

    output = multiply_power_series_by_x(b)

    output = add_power_series(a, output)

    return output
