import torch
from torch import Tensor

from ._add_laguerre_polynomial import add_laguerre_polynomial
from ._multiply_laguerre_polynomial_by_x import multiply_laguerre_polynomial_by_x


def polynomial_to_laguerre_polynomial(input: Tensor) -> Tensor:
    input = torch.atleast_1d(input)

    output = torch.zeros_like(input)

    for i in range(0, input.shape[0]):
        output = multiply_laguerre_polynomial_by_x(output, mode="same")

        output = add_laguerre_polynomial(output, torch.flip(input, dims=[0])[i])

    return output
