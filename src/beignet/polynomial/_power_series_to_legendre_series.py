import torch

from .__as_series import _as_series
from ._add_legendre_series import add_legendre_series
from ._multiply_legendre_series_by_x import multiply_legendre_series_by_x


def power_series_to_legendre_series(input):
    (input,) = _as_series([input])

    degree = len(input) - 1

    output = torch.tensor([0.0])

    for index in range(degree, -1, -1):
        output = multiply_legendre_series_by_x(output)

        output = add_legendre_series(
            output,
            torch.ravel(input[index]),
        )

    return output
