import torchaudio

from .__as_series import _as_series
from .__trim_sequence import _trim_sequence


def polymul(input, other):
    input, other = _as_series([input, other])

    output = torchaudio.functional.convolve(input, other)

    output = _trim_sequence(output)

    return output
