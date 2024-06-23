import torchaudio.functional

from .__pow import _pow


def pow_power_series(c, pow, maxpower=None):
    return _pow(torchaudio.functional.convolve, c, pow, maxpower)
