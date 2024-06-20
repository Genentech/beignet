import torch

from .__as_series import _as_series
from ._hermeadd import hermeadd
from ._hermemulx import hermemulx


def poly2herme(input):
    (input,) = _as_series([input])

    degree = len(input) - 1

    output = torch.tensor([0.0])

    for index in range(degree, -1, -1):
        output = hermemulx(output)

        output = hermeadd(
            output,
            torch.ravel(input[index]),
        )

    return output
