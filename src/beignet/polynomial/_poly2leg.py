import torch

from .__as_series import _as_series
from ._legadd import legadd
from ._legmulx import legmulx


def poly2leg(input):
    (input,) = _as_series([input])

    degree = len(input) - 1

    output = torch.tensor([0.0])

    for index in range(degree, -1, -1):
        output = legmulx(output)

        output = legadd(
            output,
            torch.ravel(input[index]),
        )

    return output
