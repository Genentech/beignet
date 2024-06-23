import torch

from .__add import _add
from .__as_series import _as_series
from ._multiply_chebyshev_series_by_x import multiply_chebyshev_series_by_x


def power_series_to_chebyshev_series(input):
    (input,) = _as_series([input])

    output = torch.tensor([0.0])

    for index in range(len(input) - 1, -1, -1):
        output = multiply_chebyshev_series_by_x(output)

        output = _add(
            output,
            torch.ravel(input[index]),
        )

    return output
