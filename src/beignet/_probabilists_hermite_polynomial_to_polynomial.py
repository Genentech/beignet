import torch
from torch import Tensor

from ._add_polynomial import add_polynomial
from ._multiply_polynomial_by_x import multiply_polynomial_by_x
from ._subtract_polynomial import subtract_polynomial


def probabilists_hermite_polynomial_to_polynomial(input: Tensor) -> Tensor:
    input = torch.atleast_1d(input)

    n = input.shape[0]

    if n == 1:
        return input

    if n == 2:
        return input
    else:
        c0 = torch.zeros_like(input)
        c0[0] = input[-2]

        c1 = torch.zeros_like(input)
        c1[0] = input[-1]

        def body(k, c0c1):
            i = n - 1 - k

            c0, c1 = c0c1

            tmp = c0

            c0 = subtract_polynomial(input[i - 2], c1 * (i - 1))

            c1 = add_polynomial(tmp, multiply_polynomial_by_x(c1, "same"))

            return c0, c1

        b = n - 2
        x = (c0, c1)
        y = x

        for index in range(0, b):
            y = body(index, y)

        c0, c1 = y

        return add_polynomial(c0, multiply_polynomial_by_x(c1, "same"))
