import torch
from torch import Tensor

from .__as_series import _as_series
from ._lagadd import lagadd
from ._lagmulx import lagmulx


def poly2lag(input: Tensor) -> Tensor:
    [input] = _as_series([input])

    output = torch.zeros_like(input)

    for i in range(0, input.shape[0]):
        output = lagmulx(output, mode="same")

        output = lagadd(output, torch.flip(input, dims=[0])[i])

    return output
