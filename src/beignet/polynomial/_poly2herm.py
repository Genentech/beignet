import torch
from torch import Tensor

from beignet.polynomial import _as_series, hermadd, hermmulx


def poly2herm(input: Tensor) -> Tensor:
    [input] = _as_series([input])

    output = torch.zeros_like(input)

    for index in range(0, input.shape[0] - 1 + 1):
        output = hermmulx(output, mode="same")

        output = hermadd(output, input[input.shape[0] - 1 - index])

    return output
