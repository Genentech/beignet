import torch
from torch import Tensor

from .__as_series import _as_series


def _trim_coefficients(
    input: Tensor,
    tol: float = 0.0,
) -> Tensor:
    if tol < 0:
        raise ValueError

    [input] = _as_series([input])

    indices = torch.nonzero(torch.abs(input) > tol)

    if indices.shape[0] == 0:
        output = input[:1] * 0
    else:
        output = input[: indices[-1] + 1]

    return output
