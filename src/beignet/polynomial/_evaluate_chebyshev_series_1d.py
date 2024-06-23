import torch
from torch import Tensor


def evaluate_chebyshev_series_1d(
    input: Tensor,
    other: Tensor,
    tensor: bool = True,
) -> Tensor:
    other = torch.ravel(other)

    if tensor:
        other = other.reshape(other.shape + (1,) * input.ndim)

    if len(other) == 1:
        c0 = other[0]
        c1 = torch.tensor([0], dtype=input.dtype)
    elif len(other) == 2:
        c0 = other[0]
        c1 = other[1]
    else:
        x2 = 2 * input
        c0 = other[-2]
        c1 = other[-1]

        for i in range(3, len(other) + 1):
            tmp = c0
            c0 = other[-i] - c1
            c1 = tmp + c1 * x2

    return c0 + c1 * input
