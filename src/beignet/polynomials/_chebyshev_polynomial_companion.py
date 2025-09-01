import math

import torch
from torch import Tensor


def chebyshev_polynomial_companion(input: Tensor) -> Tensor:
    input = torch.atleast_1d(input)

    if input.shape[0] < 2:
        raise ValueError

    if input.shape[0] == 2:
        return torch.tensor([[-input[0] / input[1]]])

    n = input.shape[0] - 1

    output = torch.zeros(
        [
            n,
            n,
        ],
        dtype=input.dtype,
    )

    scale = torch.ones([n])

    scale[1:] = math.sqrt(0.5)

    shape = output.shape

    output = torch.reshape(output, [-1])

    x = torch.full([n - 1], 1 / 2)

    x[0] = math.sqrt(0.5)

    output[1 :: n + 1] = x
    output[n :: n + 1] = x

    output = torch.reshape(output, shape)

    output[:, -1] = (
        output[:, -1] + -(input[:-1] / input[-1]) * (scale / scale[-1]) * 0.5
    )

    return output
