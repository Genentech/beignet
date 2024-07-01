import torch
from torch import Tensor

from beignet.polynomial import _as_series, chebadd, chebmulx


def poly2cheb(input: Tensor) -> Tensor:
    [input] = _as_series([input])

    output = torch.zeros_like(input)

    for i in range(0, input.shape[0] - 1 + 1):
        output = chebmulx(output, mode="same")

        output = chebadd(output, input[input.shape[0] - 1 - i])

    return output
