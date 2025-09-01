import torch
from torch import Tensor

from ._probabilists_hermite_polynomial_companion import (
    probabilists_hermite_polynomial_companion,
)


def probabilists_hermite_polynomial_roots(input: Tensor) -> Tensor:
    input = torch.atleast_1d(input)

    if input.shape[0] <= 1:
        return torch.tensor([], dtype=input.dtype)

    if input.shape[0] == 2:
        return torch.tensor([-input[0] / input[1]])

    output = probabilists_hermite_polynomial_companion(input)

    output = torch.flip(output, dims=[0])
    output = torch.flip(output, dims=[1])

    output = torch.linalg.eigvals(output)

    output, _ = torch.sort(output.real)

    return output
