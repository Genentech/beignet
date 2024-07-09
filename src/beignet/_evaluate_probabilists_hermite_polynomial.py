import torch
from torch import Tensor


def evaluate_probabilists_hermite_polynomial(
    input: Tensor,
    coefficients: Tensor,
    tensor: bool = True,
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

            for i in range(3, coefficients.shape[0] + 1):
                previous = a

                size = size - 1

                a = coefficients[-i] - b * (size - 1.0)

                b = previous + b * input

    return a + b * input
