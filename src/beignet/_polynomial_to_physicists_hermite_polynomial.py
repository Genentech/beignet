import torch
from torch import Tensor

from ._add_physicists_hermite_polynomial import add_physicists_hermite_polynomial
from ._multiply_physicists_hermite_polynomial_by_x import (
    multiply_physicists_hermite_polynomial_by_x,
)


def polynomial_to_physicists_hermite_polynomial(input: Tensor) -> Tensor:
    input = torch.atleast_1d(input)

    output = torch.zeros_like(input)

    for index in range(0, input.shape[0] - 1 + 1):
        output = multiply_physicists_hermite_polynomial_by_x(output, mode="same")

        output = add_physicists_hermite_polynomial(
            output, input[input.shape[0] - 1 - index]
        )

    return output
