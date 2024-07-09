import torch
from torch import Tensor

from ._add_polynomial import add_polynomial
from ._multiply_polynomial_by_x import multiply_polynomial_by_x
from ._subtract_polynomial import subtract_polynomial


def legendre_polynomial_to_polynomial(input: Tensor) -> Tensor:
    input = torch.atleast_1d(input)

    n = input.shape[0]

    if n < 3:
        return input

    c0 = torch.zeros_like(input)
    c0[0] = input[-2]

    c1 = torch.zeros_like(input)
    c1[0] = input[-1]

    def body(k, c0c1):
        i = n - 1 - k

        c0, c1 = c0c1

        tmp = c0

        c0 = subtract_polynomial(input[i - 2], c1 * (i - 1) / i)

        c1 = add_polynomial(tmp, multiply_polynomial_by_x(c1, "same") * (2 * i - 1) / i)

        return c0, c1

    x = (c0, c1)

    for i in range(0, n - 2):
        x = body(i, x)

    c0, c1 = x

    output = multiply_polynomial_by_x(c1, "same")

    output = add_polynomial(c0, output)

    return output
