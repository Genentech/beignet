import math

import torchaudio.functional
from torch import Tensor

from .__as_series import _as_series
from .__c_series_to_z_series import _c_series_to_z_series


def multiply_chebyshev_series(input: Tensor, other: Tensor) -> Tensor:
    input, other = _as_series([input, other])

    input = _c_series_to_z_series(input)
    other = _c_series_to_z_series(other)

    output = torchaudio.functional.convolve(input, other)

    n = (math.prod(output.shape) + 1) // 2

    output = output[n - 1 :]

    output[1:n] = output[1:n] * 2

    if len(output) != 0 and output[-1] == 0:
        for index in range(len(output) - 1, -1, -1):
            if output[index] != 0:
                break

        output = output[: index + 1]

    return output
