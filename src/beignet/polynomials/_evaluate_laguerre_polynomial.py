import torch
from torch import Tensor


def evaluate_laguerre_polynomial(
    input: Tensor, coefficients: Tensor, tensor: bool = True
):
    coefficients = torch.atleast_1d(coefficients)

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
            size = coefficients.shape[0]

            a = coefficients[-2] * torch.ones_like(input)
            b = coefficients[-1] * torch.ones_like(input)

            for index in range(3, coefficients.shape[0] + 1):
                previous = a

                size = size - 1

                a = coefficients[-index] - (b * (size - 1.0)) / size

                b = previous + (b * ((2.0 * size - 1.0) - input)) / size

    return a + b * (1.0 - input)
