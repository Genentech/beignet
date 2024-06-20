import torch

from .__as_series import _as_series
from ._add_physicists_hermite_series import add_physicists_hermite_series
from ._hermmulx import hermmulx


def power_series_to_physicists_hermite_series(input):
    (input,) = _as_series([input])

    degree = len(input) - 1

    output = torch.tensor([0.0])

    for index in range(degree, -1, -1):
        output = hermmulx(output)

        output = add_physicists_hermite_series(
            output,
            torch.ravel(input[index]),
        )

    return output
