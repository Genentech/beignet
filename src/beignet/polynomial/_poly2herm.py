import torch
from torch import Tensor

from .__as_series import _as_series
from ._hermadd import hermadd
from ._hermmulx import hermmulx


def poly2herm(input: Tensor) -> Tensor:
    [input] = _as_series([input])

    output = torch.zeros_like(input)

    for index in range(0, input.shape[0] - 1 + 1):
        output = hermmulx(output, mode="same")

        output = hermadd(output, input[input.shape[0] - 1 - index])

    return output
