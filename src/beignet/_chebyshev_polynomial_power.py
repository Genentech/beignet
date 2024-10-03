import math

import torch
from torch import Tensor
from ._convolve import convolve

from ._add_chebyshev_polynomial import add_chebyshev_polynomial


def chebyshev_polynomial_power(
    input: Tensor,
    exponent: float | Tensor,
    maximum_exponent: float | Tensor = 16.0,
) -> Tensor:
    input = torch.atleast_1d(input)

    _exponent = int(exponent)

    if _exponent != exponent or _exponent < 0:
        raise ValueError

    if maximum_exponent is not None and _exponent > maximum_exponent:
        raise ValueError

    match _exponent:
        case 0:
            output = torch.tensor([1.0], dtype=input.dtype)
        case 1:
            output = input
        case _:
            output = torch.zeros(input.shape[0] * exponent, dtype=input.dtype)

            output = add_chebyshev_polynomial(output, input)

            index1 = math.prod(input.shape)
            output2 = torch.zeros(2 * index1 - 1, dtype=input.dtype)
            output2[index1 - 1 :] = input / 2.0
            output2 = torch.flip(output2, dims=[0]) + output2
            zs = output2

            index = math.prod(output.shape)
            output1 = torch.zeros(2 * index - 1, dtype=output.dtype)
            output1[index - 1 :] = output / 2.0
            output1 = torch.flip(output1, dims=[0]) + output1
            output = output1

            for _ in range(2, _exponent + 1):
                output = convolve(output, zs, mode="same")

            n = (math.prod(output.shape) + 1) // 2
            c = output[n - 1 :]
            c[1:n] = c[1:n] * 2.0
            output = c

    return output
