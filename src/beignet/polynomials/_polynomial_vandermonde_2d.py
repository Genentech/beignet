import functools
import operator

import torch
from torch import Tensor

from ._polynomial_vandermonde import polynomial_vandermonde


def polynomial_vandermonde_2d(x: Tensor, y: Tensor, degree: Tensor) -> Tensor:
    r"""
    Parameters
    ----------
    x : Tensor

    y : Tensor

    degree : Tensor

    Returns
    -------
    output : Tensor
    """
    functions = (
        polynomial_vandermonde,
        polynomial_vandermonde,
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
