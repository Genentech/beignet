import torch
from torch import Tensor

from .__as_series import _as_series


def multiply_laguerre_series_by_x(input: Tensor) -> Tensor:
    (input,) = _as_series([input])

    if len(input) == 1 and input[0] == 0:
        return input

    output = torch.empty(len(input) + 1, dtype=input.dtype)

    output[0] = input[0]
    output[1] = -input[0]

    for index in range(1, len(input)):
        output[index + 1] = -input[index] * (index + 1)

        output[index] = output[index] + (input[index] * (2 * index + 1))

        output[index - 1] = output[index - 1] - (input[index] * index)

    return output
