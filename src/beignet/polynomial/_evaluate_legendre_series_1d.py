import torch
from torch import Tensor


def evaluate_legendre_series_1d(
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
        nd = len(other)
        c0 = other[-2]
        c1 = other[-1]
        for i in range(3, len(other) + 1):
            tmp = c0
            nd = nd - 1
            c0 = other[-i] - (c1 * (nd - 1)) / nd
            c1 = tmp + (c1 * input * (2 * nd - 1)) / nd
    return c0 + c1 * input
