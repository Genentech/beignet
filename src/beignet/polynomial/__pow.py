import torch
from torch import Tensor

from .__as_series import _as_series


def _pow(func, input: Tensor, exponent, maximum_exponent) -> Tensor:
    (input,) = _as_series([input])

    exponent = int(exponent)

    if exponent != exponent or exponent < 0:
        raise ValueError

    if maximum_exponent is not None and exponent > maximum_exponent:
        raise ValueError

    if exponent == 0:
        return torch.tensor([1], dtype=input.dtype)

    if exponent == 1:
        return input

    output = input

    for _ in range(2, exponent + 1):
        output = func(output, input)

    return output
