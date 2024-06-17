import torch
from torch import Tensor


def evaluate_polynomial_from_roots(
    input: Tensor,
    other: Tensor,
    tensor: bool = True,
) -> Tensor:
    if other.ndim == 0:
        other = torch.ravel(other)

    if tensor:
        other = torch.reshape(other, other.shape + (1,) * input.ndim)

    if input.ndim >= other.ndim:
        raise ValueError

    output = torch.prod(input - other, dim=0)

    return output
