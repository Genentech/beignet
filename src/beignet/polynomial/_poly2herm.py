import torch

from .__as_series import _as_series
from ._hermadd import hermadd
from ._hermmulx import hermmulx


def poly2herm(input):
    (input,) = _as_series([input])

    degree = len(input) - 1

    output = torch.tensor([0.0])

    for index in range(degree, -1, -1):
        output = hermmulx(output)

        output = hermadd(
            output,
            torch.ravel(input[index]),
        )

    return output
