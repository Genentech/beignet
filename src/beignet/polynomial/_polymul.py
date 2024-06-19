import torchaudio

from beignet.polynomial import _as_series, _trim_sequence


def polymul(input, other):
    input, other = _as_series([input, other])

    output = torchaudio.functional.convolve(input, other)

    output = _trim_sequence(output)

    return output
