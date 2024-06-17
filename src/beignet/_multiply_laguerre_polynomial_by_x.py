from typing import Literal

import torch
from torch import Tensor


def multiply_laguerre_polynomial_by_x(
    input: Tensor,
    mode: Literal["full", "same", "valid"] = "full",
) -> Tensor:
    input = torch.atleast_1d(input)

    output = torch.zeros(input.shape[0] + 1, dtype=input.dtype)

    output[0] = +input[0]
    output[1] = -input[0]

    i = torch.arange(1, input.shape[0])

    output[i + 1] = -input[i] * (i + 1)

    output[i] = output[i] + input[i] * (2 * i + 1)

    output[i - 1] = output[i - 1] - input[i] * i

    if mode == "same":
        output = output[: input.shape[0]]

    return output
