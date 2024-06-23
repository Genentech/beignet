import torch
from torch import Tensor

from .__as_series import _as_series


def multiply_laguerre_series_by_x(input: Tensor) -> Tensor:
    (input,) = _as_series([input])

    if len(input) == 1 and input[0] == 0:
        return input

    prd = torch.empty(len(input) + 1, dtype=input.dtype)
    prd[0] = input[0]
    prd[1] = -input[0]
    for i in range(1, len(input)):
        prd[i + 1] = -input[i] * (i + 1)
        prd[i] += input[i] * (2 * i + 1)
        prd[i - 1] -= input[i] * i
    return prd
