from typing import Literal

import torch
from torch import Tensor

from .__as_series import _as_series


def hermmulx(
    input: Tensor,
    mode: Literal["full", "same", "valid"] = "full",
) -> Tensor:
    [input] = _as_series([input])

    output = torch.zeros(input.shape[0] + 1, dtype=input.dtype)

    output[1] = input[0] / 2.0

    i = torch.arange(1, input.shape[0])

    output[i + 1] = input[i] / 2.0
    output[i - 1] = output[i - 1] + input[i] * i

    if mode == "same":
        output = output[: input.shape[0]]

    return output
