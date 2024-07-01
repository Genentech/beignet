from typing import Literal

import torch
from torch import Tensor

from beignet.polynomial import _as_series


def chebmulx(
    input: Tensor,
    mode: Literal["full", "same", "valid"] = "full",
) -> Tensor:
    [input] = _as_series([input])

    output = torch.zeros(input.shape[0] + 1, dtype=input.dtype)

    output[1] = input[0]

    if input.shape[0] > 1:
        output[2:] = input[1:] / 2

        output[0:-2] = output[0:-2] + input[1:] / 2

    if mode == "same":
        output = output[: input.shape[0]]

    return output
