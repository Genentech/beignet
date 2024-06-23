import torch
import torchaudio
from torch import Tensor

from .__as_series import _as_series
from .__c_series_to_z_series import _c_series_to_z_series
from .__z_series_to_c_series import _z_series_to_c_series


def pow_chebyshev_series(
    input: Tensor,
    exponent: Tensor,
    maximum_exponent: Tensor = 16,
) -> Tensor:
    (input,) = _as_series([input])

    power = int(exponent)

    if power != exponent or power < 0:
        raise ValueError

    if maximum_exponent is not None and power > maximum_exponent:
        raise ValueError

    if power == 0:
        return torch.tensor([1], dtype=input.dtype)

    if power == 1:
        return input

    z_series = _c_series_to_z_series(input)

    output = z_series

    for _ in range(2, power + 1):
        output = torchaudio.functional.convolve(output, z_series)

    output = _z_series_to_c_series(output)

    return output
