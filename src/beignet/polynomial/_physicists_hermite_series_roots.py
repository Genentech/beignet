import torch

from .__as_series import _as_series
from ._physicists_hermite_series_companion import physicists_hermite_series_companion


def physicists_hermite_series_roots(input):
    (input,) = _as_series([input])

    if len(input) <= 1:
        return torch.tensor([], dtype=input.dtype)

    if len(input) == 2:
        return torch.tensor([-0.5 * input[0] / input[1]], dtype=input.dtype)

    output = physicists_hermite_series_companion(input)

    output = torch.flip(output, dims=[0])
    output = torch.flip(output, dims=[1])

    output = torch.linalg.eigvals(output)

    output, _ = torch.sort(output)

    return output
