import torch
from torch import Tensor

from beignet.polynomial import _as_series, legadd, legmulx


def poly2leg(input: Tensor) -> Tensor:
    [input] = _as_series([input])

    output = torch.zeros_like(input)

    for i in range(0, input.shape[0] - 1 + 1):
        output = legmulx(output, mode="same")

        output = legadd(output, input[input.shape[0] - 1 - i])

    return output
