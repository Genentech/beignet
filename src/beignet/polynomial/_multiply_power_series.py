import torchaudio
from torch import Tensor

from .__as_series import _as_series
from .__trim_sequence import _trim_sequence


def multiply_power_series(input: Tensor, other: Tensor) -> Tensor:
    input, other = _as_series([input, other])

    output = torchaudio.functional.convolve(input, other)

    output = _trim_sequence(output)

    return output
