import torch
from torch import Tensor

from ._legendre_polynomial_companion import legendre_polynomial_companion


def legendre_polynomial_roots(input: Tensor) -> Tensor:
    input = torch.atleast_1d(input)

    if input.shape[0] <= 1:
        return torch.tensor([], dtype=input.dtype)

    if input.shape[0] == 2:
        return torch.tensor([-input[0] / input[1]])

    output = legendre_polynomial_companion(input)

    output = torch.flip(output, dims=[0])
    output = torch.flip(output, dims=[1])

    output = torch.linalg.eigvals(output)

    output, _ = torch.sort(output.real)

    return output
