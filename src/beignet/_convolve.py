import math
from typing import Literal

import torch
import torch.nn.functional
from torch import Tensor


def convolve(
    input: Tensor,
    other: Tensor,
    mode: Literal["full", "same", "valid"] = "full",
) -> Tensor:
    if input.ndim != other.ndim:
        raise ValueError

    for i in range(input.ndim - 1):
        a = input.size(i)
        b = other.size(i)

        if a == b or a == 1 or b == 1:
            continue

        raise ValueError

    if mode not in {"full", "same", "valid"}:
        raise ValueError

    x_size, y_size = input.shape[-1], other.shape[-1]

    if input.shape[-1] < other.shape[-1]:
        input, other = other, input

    if input.shape[:-1] != other.shape[:-1]:
        input_shape = []

        for i, j in zip(input.shape[:-1], other.shape[:-1], strict=False):
            input_shape = [*input_shape, max(i, j)]

        input = torch.broadcast_to(
            input,
            [*input_shape, input.shape[-1]],
        )

        other = torch.broadcast_to(
            other,
            [*input_shape, other.shape[-1]],
        )

    input = torch.reshape(
        input,
        [math.prod(input.shape[:-1]), input.shape[-1]],
    )

    other = torch.unsqueeze(
        torch.flip(
            torch.reshape(
                other,
                [math.prod(input.shape[:-1]), other.shape[-1]],
            ),
            dims=[-1],
        ),
        dim=1,
    )

    output = torch.reshape(
        torch.nn.functional.conv1d(
            input,
            other,
            groups=input.shape[0],
            padding=other.shape[-1] - 1,
        ),
        [*input.shape[:-1], -1],
    )

    match mode:
        case "same":
            size = x_size

            m = (output.shape[-1] - size) // 2
            n = m + size

            output = output[..., m:n]
        case "valid":
            size = max(x_size, y_size) - min(x_size, y_size) + 1

            m = (output.shape[-1] - size) // 2
            n = m + size

            output = output[..., m:n]

    return output
