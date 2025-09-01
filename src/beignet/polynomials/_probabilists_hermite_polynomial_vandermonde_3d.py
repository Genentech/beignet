import functools
import operator

import torch
from torch import Tensor

from ._probabilists_hermite_polynomial_vandermonde import (
    probabilists_hermite_polynomial_vandermonde,
)


def probabilists_hermite_polynomial_vandermonde_3d(
    x: Tensor,
    y: Tensor,
    z: Tensor,
    degree: Tensor,
) -> Tensor:
    functions = (
        probabilists_hermite_polynomial_vandermonde,
        probabilists_hermite_polynomial_vandermonde,
        probabilists_hermite_polynomial_vandermonde,
    )

    n = len(functions)

    if n != len([x, y, z]):
        raise ValueError

    if n != len(degree):
        raise ValueError

    if n == 0:
        raise ValueError

    matrices = []

    for i in range(n):
        matrix = functions[i]((x, y, z)[i], degree[i])

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
