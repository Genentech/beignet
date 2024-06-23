import torch
from torch import Tensor

from .__as_series import _as_series
from ._add_power_series import add_power_series
from ._multiply_power_series_by_x import multiply_power_series_by_x
from ._subtract_power_series import subtract_power_series


def laguerre_series_to_power_series(input: Tensor) -> Tensor:
    (input,) = _as_series([input])

    n = len(input)

    if n == 1:
        return input

    a = torch.ravel(input[-2])
    b = torch.ravel(input[-1])

    for index in range(n - 1, 1, -1):
        c = a

        a = subtract_power_series(
            torch.ravel(input[index - 2]),
            (b * (index - 1)) / index,
        )

        b = subtract_power_series(
            (2 * index - 1) * b,
            multiply_power_series_by_x(b),
        )

        b = b / index

        b = add_power_series(c, b)

    return add_power_series(
        a,
        subtract_power_series(
            b,
            multiply_power_series_by_x(b),
        ),
    )
