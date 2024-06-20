import torch

from .__as_series import _as_series
from ._lagadd import lagadd
from ._lagmulx import lagmulx


def poly2lag(input):
    (input,) = _as_series([input])

    output = torch.tensor([0.0])

    for index in torch.flip(input, dims=[0]):
        output = lagmulx(output)

        output = lagadd(
            output,
            torch.ravel(index),
        )

    return output
