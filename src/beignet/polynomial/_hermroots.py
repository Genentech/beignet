import torch
from torch import Tensor

from beignet.polynomial import _as_series, hermcompanion


def hermroots(input: Tensor) -> Tensor:
    [input] = _as_series([input])

    if input.shape[0] <= 1:
        return torch.tensor([], dtype=input.dtype)

    if input.shape[0] == 2:
        return torch.tensor([-0.5 * input[0] / input[1]])

    output = hermcompanion(input)

    output = torch.flip(output, dims=[0])
    output = torch.flip(output, dims=[1])

    output = torch.linalg.eigvals(output)

    output, _ = torch.sort(output.real)

    return output
