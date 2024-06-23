import torch
from torch import Tensor

from beignet.polynomial._legendre_series_companion import legendre_series_companion

from .__as_series import _as_series


def legendre_series_roots(input: Tensor) -> Tensor:
    (input,) = _as_series([input])

    if len(input) < 2:
        return torch.tensor([], dtype=input.dtype)

    if len(input) == 2:
        return torch.tensor([-input[0] / input[1]], dtype=input.dtype)

    output = legendre_series_companion(input)

    output = torch.flip(output, dims=[0])
    output = torch.flip(output, dims=[1])

    output = torch.linalg.eigvals(output)

    output, _ = torch.sort(output)

    return output
