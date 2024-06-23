import torch
from torch import Tensor


def evaluate_power_series_1d(
    input: Tensor,
    other: Tensor,
    tensor: bool = True,
) -> Tensor:
    other = torch.ravel(other)

    if tensor:
        other = torch.reshape(other, other.shape + (1,) * input.ndim)

    c0 = other[-1] + input * 0

    for i in range(2, len(other) + 1):
        c0 = other[-i] + c0 * input

    return c0
