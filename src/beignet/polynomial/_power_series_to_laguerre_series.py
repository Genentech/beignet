import torch

from .__as_series import _as_series
from ._add_laguerre_series import add_laguerre_series
from ._lagmulx import lagmulx


def power_series_to_laguerre_series(input):
    (input,) = _as_series([input])

    output = torch.tensor([0.0])

    for index in torch.flip(input, dims=[0]):
        output = lagmulx(output)

        output = add_laguerre_series(
            output,
            torch.ravel(index),
        )

    return output
