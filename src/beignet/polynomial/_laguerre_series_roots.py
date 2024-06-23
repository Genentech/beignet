import torch

from beignet.polynomial._laguerre_series_companion import laguerre_series_companion

from .__as_series import _as_series


def laguerre_series_roots(input):
    (input,) = _as_series([input])

    if len(input) <= 1:
        return torch.tensor([], dtype=input.dtype)

    if len(input) == 2:
        return torch.tensor([1 + input[0] / input[1]])

    output = laguerre_series_companion(input)

    output = output[::-1, ::-1]

    output = torch.linalg.eigvals(output)

    output, _ = torch.sort(output)

    return output
