import torch

from .__as_series import _as_series
from ._add_probabilists_hermite_series import add_probabilists_hermite_series
from ._multiply_probabilists_hermite_series_by_x import (
    multiply_probabilists_hermite_series_by_x,
)


def power_series_to_probabilists_hermite_series(input):
    (input,) = _as_series([input])

    degree = len(input) - 1

    output = torch.tensor([0.0])

    for index in range(degree, -1, -1):
        output = multiply_probabilists_hermite_series_by_x(output)

        output = add_probabilists_hermite_series(
            output,
            torch.ravel(input[index]),
        )

    return output
