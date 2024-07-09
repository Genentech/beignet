import functools
import operator

import torch
from torch import Tensor

from ._polynomial_vandermonde import polynomial_vandermonde


def polynomial_vandermonde_3d(
    x: Tensor, y: Tensor, z: Tensor, degree: Tensor
) -> Tensor:
    r"""
    Parameters
    ----------
    x : Tensor

    y : Tensor

    z : Tensor

    degree : Tensor

    Returns
    -------
    output : Tensor
    """
    vandermonde_functions = (
        polynomial_vandermonde,
        polynomial_vandermonde,
        polynomial_vandermonde,
    )

    n = len(vandermonde_functions)

    if n != len([x, y, z]):
        raise ValueError

    if n != len(degree):
        raise ValueError

    if n == 0:
        raise ValueError

    matrices = []

    for i in range(n):
        matrix = vandermonde_functions[i]((x, y, z)[i], degree[i])

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
