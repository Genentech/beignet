import torch
from torch import Tensor

from .__as_series import _as_series
from ._add_power_series import add_power_series
from ._multiply_power_series_by_x import multiply_power_series_by_x
from ._subtract_power_series import subtract_power_series


def chebyshev_series_to_power_series(input: Tensor) -> Tensor:
    (input,) = _as_series([input])

    n = len(input)

    if n < 3:
        return input

    a = torch.ravel(input[-2])
    b = torch.ravel(input[-1])

    for index in range(n - 1, 1, -1):
        c = a

        a = subtract_power_series(
            torch.ravel(input[index - 2]),
            b,
        )

        b = add_power_series(
            c,
            multiply_power_series_by_x(b) * 2,
        )

    return add_power_series(
        a,
        multiply_power_series_by_x(b),
    )
