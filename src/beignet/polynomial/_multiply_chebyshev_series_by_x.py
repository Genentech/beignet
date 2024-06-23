import torch
from torch import Tensor

from .__as_series import _as_series


def multiply_chebyshev_series_by_x(input: Tensor) -> Tensor:
    [input] = _as_series([input])

    if len(input) == 1 and input[0] == 0:
        return input

    output = torch.empty(len(input) + 1, dtype=input.dtype)

    output[0] = input[0] * 0
    output[1] = input[0]

    if len(input) > 1:
        tmp = input[1:] / 2

        output[2:] = tmp
        output[0:-2] += tmp

    return output
