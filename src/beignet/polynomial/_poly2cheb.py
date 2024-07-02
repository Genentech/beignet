import torch
from torch import Tensor

from .__as_series import _as_series
from ._chebadd import chebadd
from ._chebmulx import chebmulx


def poly2cheb(input: Tensor) -> Tensor:
    [input] = _as_series([input])

    output = torch.zeros_like(input)

    for i in range(0, input.shape[0] - 1 + 1):
        output = chebmulx(output, mode="same")

        output = chebadd(output, input[input.shape[0] - 1 - i])

    return output
