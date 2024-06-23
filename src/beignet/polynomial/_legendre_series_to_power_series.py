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
    else:
        c0 = input[-2]
        c1 = input[-1]

        for i in range(n - 1, 1, -1):
            tmp = c0
            c0 = subtract_power_series(input[i - 2], (c1 * (i - 1)) / i)
            c1 = add_power_series(
                tmp, (multiply_power_series_by_x(c1) * (2 * i - 1)) / i
            )
        return add_power_series(c0, multiply_power_series_by_x(c1))
