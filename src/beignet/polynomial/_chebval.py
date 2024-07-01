import torch
from torch import Tensor

from beignet.polynomial import _as_series


def chebval(
    input: Tensor,
    coefficients: Tensor,
    tensor: bool = True,
) -> Tensor:
    [coefficients] = _as_series([coefficients])

    if tensor:
        coefficients = torch.reshape(
            coefficients,
            coefficients.shape + (1,) * input.ndim,
        )

    match coefficients.shape[0]:
        case 1:
            a = coefficients[0]
            b = 0.0
        case 2:
            a = coefficients[0]
            b = coefficients[1]
        case _:
            a = coefficients[-2] * torch.ones_like(input)
            b = coefficients[-1] * torch.ones_like(input)

            for i in range(3, coefficients.shape[0] + 1):
                previous = a

                a = coefficients[-i] - b
                b = previous + b * 2.0 * input

    return a + b * input
