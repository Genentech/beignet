import math

from torch import Tensor


def _z_series_to_c_series(
    input: Tensor,
) -> Tensor:
    n = (math.prod(input.shape) + 1) // 2

    c = input[n - 1 :]

    c[1:n] = c[1:n] * 2.0

    return c
