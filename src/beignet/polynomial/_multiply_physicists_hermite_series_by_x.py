import torch
from torch import Tensor

from .__as_series import _as_series


def multiply_physicists_hermite_series_by_x(input: Tensor) -> Tensor:
    (input,) = _as_series([input])

    if len(input) == 1 and input[0] == 0:
        return input

    output = torch.empty(len(input) + 1, dtype=input.dtype)

    output[0] = input[0] * 0
    output[1] = input[0] / 2

    for i in range(1, len(input)):
        output[i + 1] = input[i] / 2
        output[i - 1] += input[i] * i

    return output
