import math

from torch import Tensor


def _z_series_to_c_series(input: Tensor) -> Tensor:
    index = (math.prod(input.shape) + 1) // 2

    output = input[index - 1 :]

    output[1:index] = output[1:index] * 2

    return output
