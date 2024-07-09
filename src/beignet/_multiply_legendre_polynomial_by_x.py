from typing import Literal

import torch
from torch import Tensor


def multiply_legendre_polynomial_by_x(
    input: Tensor, mode: Literal["full", "same"] = "full"
) -> Tensor:
    input = torch.atleast_1d(input)

    output = torch.zeros(input.shape[0] + 1, dtype=input.dtype)
    output[1] = input[0]

    for index in range(1, input.shape[0]):
        output[index + 1] = (input[index] * (index + 1)) / (index + index + 1)
        output[index - 1] = output[index - 1] + (input[index] * (index + 0)) / (
            index + index + 1
        )

    if mode == "same":
        output = output[: input.shape[0]]

    return output
