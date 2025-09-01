import functools
import operator

import torch
from torch import Tensor

from ._physicists_hermite_polynomial_vandermonde import (
    physicists_hermite_polynomial_vandermonde,
)


def physicists_hermite_polynomial_vandermonde_2d(
    x: Tensor,
    y: Tensor,
    degree: Tensor,
) -> Tensor:
    functions = (
        physicists_hermite_polynomial_vandermonde,
        physicists_hermite_polynomial_vandermonde,
    )

    n = len(functions)

    if n != len([x, y]):
        raise ValueError

    if n != len(degree):
        raise ValueError

    if n == 0:
        raise ValueError

    matrices = []

    for i in range(n):
        matrix = functions[i]((x, y)[i], degree[i])

        matrices = [
            *matrices,
            matrix[(..., *tuple(slice(None) if j == i else None for j in range(n)))],
        ]

    vandermonde = functools.reduce(
        operator.mul,
        matrices,
    )

    return torch.reshape(
        vandermonde,
        [*vandermonde.shape[: -len(degree)], -1],
    )
