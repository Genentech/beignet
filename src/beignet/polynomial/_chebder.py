import torch
from torch import Tensor

from beignet.polynomial import _as_series


def chebder(
    input: Tensor,
    order=1,
    scale=1,
    axis=0,
) -> Tensor:
    if order < 0:
        raise ValueError

    [input] = _as_series([input])

    if order == 0:
        return input

    output = torch.moveaxis(input, axis, 0)

    n = output.shape[0]

    if order >= n:
        output = torch.zeros_like(output[:1])
    else:
        for _ in range(order):
            n = n - 1

            output = output * scale

            derivative = torch.empty((n,) + output.shape[1:], dtype=output.dtype)

            for i in range(0, n - 2):
                j = n - i

                derivative[j - 1] = (2 * j) * output[j]

                output = output.at[j - 2].add((j * output[j]) / (j - 2))

            if n > 1:
                derivative[1] = 4 * output[2]

            derivative[0] = output[1]

    return torch.moveaxis(output, 0, axis)
