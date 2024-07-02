import torch
import torchaudio.functional
from torch import Tensor

from .__as_series import _as_series
from .__c_series_to_z_series import _c_series_to_z_series
from .__z_series_to_c_series import _z_series_to_c_series
from ._chebadd import chebadd


def chebpow(
    input: Tensor,
    exponent: float | Tensor,
    maximum_exponent: float | Tensor = 16.0,
) -> Tensor:
    [input] = _as_series([input])

    _exponent = int(exponent)

    if _exponent != exponent or _exponent < 0:
        raise ValueError

    if maximum_exponent is not None and _exponent > maximum_exponent:
        raise ValueError

    match _exponent:
        case 0:
            output = torch.tensor([1.0], dtype=input.dtype)
        case 1:
            output = input
        case _:
            output = torch.zeros(input.shape[0] * exponent, dtype=input.dtype)

            output = chebadd(output, input)

            zs = _c_series_to_z_series(input)

            output = _c_series_to_z_series(output)

            for _ in range(2, _exponent + 1):
                output = torchaudio.functional.convolve(output, zs, mode="same")

            output = _z_series_to_c_series(output)

    return output
