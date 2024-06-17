import torch
from torch import Tensor

from ._add_probabilists_hermite_polynomial import add_probabilists_hermite_polynomial
from ._multiply_probabilists_hermite_polynomial_by_x import (
    multiply_probabilists_hermite_polynomial_by_x,
)


def polynomial_to_probabilists_hermite_polynomial(input: Tensor) -> Tensor:
    input = torch.atleast_1d(input)

    output = torch.zeros_like(input)

    for i in range(0, input.shape[0] - 1 + 1):
        output = multiply_probabilists_hermite_polynomial_by_x(output, mode="same")

        output = add_probabilists_hermite_polynomial(
            output, input[input.shape[0] - 1 - i]
        )

    return output
