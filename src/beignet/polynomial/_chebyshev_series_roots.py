import torch
from torch import Tensor

from .__as_series import _as_series
from ._chebyshev_series_companion import chebyshev_series_companion


def chebyshev_series_roots(input: Tensor) -> Tensor:
    (input,) = _as_series([input])

    if len(input) < 2:
        return torch.tensor([], dtype=input.dtype)

    if len(input) == 2:
        return torch.tensor([-input[0] / input[1]], dtype=input.dtype)

    output = chebyshev_series_companion(input)

    output = torch.flip(output, dims=[0])
    output = torch.flip(output, dims=[1])

    output = torch.linalg.eigvals(output)

    output, _ = torch.sort(output)

    return output
