import math

import torch
from torch import Tensor

from ._chebyshev_nodes_1 import chebyshev_nodes_1
from ._chebyshev_series_vandermonde_1d import chebyshev_series_vandermonde_1d


def chebyshev_interpolation(
    func: callable,
    degree: int,
    args: tuple = (),
) -> Tensor:
    degree = torch.tensor(degree)

    if (
        degree.ndim > 0
        or degree.dtype
        not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
            torch.uint16,
            torch.uint32,
            torch.uint64,
        }
        or math.prod(degree.shape) == 0
    ):
        raise TypeError

    if degree < 0:
        raise ValueError

    output = torch.dot(
        chebyshev_series_vandermonde_1d(
            chebyshev_nodes_1(degree + 1),
            degree,
        ).T,
        func(
            chebyshev_nodes_1(degree + 1),
            *args,
        ),
    )

    output[0] = output[0] / (degree + 1)

    output[1:] = output[1:] / (0.5 * (degree + 1))

    return output
