import torch
from torch import Tensor

from ._add_polynomial import add_polynomial
from ._multiply_polynomial_by_x import multiply_polynomial_by_x
from ._subtract_polynomial import subtract_polynomial


def chebyshev_polynomial_to_polynomial(input: Tensor) -> Tensor:
    input = torch.atleast_1d(input)

    n = input.shape[0]

    if n < 3:
        return input

    c0 = torch.zeros_like(input)
    c0[0] = input[-2]

    c1 = torch.zeros_like(input)
    c1[0] = input[-1]

    for index in range(0, n - 2):
        i1 = n - 1 - index

        tmp = c0

        c0 = subtract_polynomial(input[i1 - 2], c1)

        c1 = add_polynomial(tmp, multiply_polynomial_by_x(c1, "same") * 2)

    output = multiply_polynomial_by_x(c1, "same")

    output = add_polynomial(c0, output)

    return output
