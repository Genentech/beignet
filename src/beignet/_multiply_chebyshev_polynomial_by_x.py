from typing import Literal

import torch
from torch import Tensor


def multiply_chebyshev_polynomial_by_x(
    input: Tensor,
    mode: Literal["full", "same", "valid"] = "full",
) -> Tensor:
    input = torch.atleast_1d(input)

    output = torch.zeros(input.shape[0] + 1, dtype=input.dtype)

    output[1] = input[0]

    if input.shape[0] > 1:
        output[2:] = input[1:] / 2

        output[0:-2] = output[0:-2] + input[1:] / 2

    if mode == "same":
        output = output[: input.shape[0]]

    return output
