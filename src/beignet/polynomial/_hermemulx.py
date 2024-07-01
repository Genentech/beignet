from typing import Literal

import torch
from torch import Tensor

from beignet.polynomial import _as_series


def hermemulx(
    input: Tensor,
    mode: Literal["full", "same", "valid"] = "full",
) -> Tensor:
    [input] = _as_series([input])

    output = torch.zeros(input.shape[0] + 1, dtype=input.dtype)

    output[1] = input[0]

    index = torch.arange(1, input.shape[0])

    output[index + 1] = input[index]
    output[index - 1] = output[index - 1] + input[index] * index

    if mode == "same":
        output = output[: input.shape[0]]

    return output
