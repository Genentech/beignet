import torch
from torch import Tensor

from .__as_series import _as_series
from ._lagcompanion import lagcompanion


def lagroots(input: Tensor) -> Tensor:
    [input] = _as_series([input])

    if input.shape[0] <= 1:
        return torch.tensor([], dtype=input.dtype)

    if input.shape[0] == 2:
        return torch.tensor([1 + input[0] / input[1]])

    output = lagcompanion(input)

    output = torch.flip(output, dims=[0])
    output = torch.flip(output, dims=[1])

    output = torch.linalg.eigvals(output)

    output, _ = torch.sort(output.real)

    return output
