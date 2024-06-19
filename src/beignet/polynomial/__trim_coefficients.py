import torch
from torch import Tensor

from .__as_series import _as_series


def _trim_coefficients(input: Tensor, tolerance: float = 0.0) -> Tensor:
    if tolerance < 0:
        raise ValueError

    (input,) = _as_series([input])

    indices = torch.nonzero(torch.abs(input) > tolerance)

    if len(indices) == 0:
        return torch.zeros_like(input[:1])
    else:
        return input[: indices[-1] + 1]
