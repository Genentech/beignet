import torchaudio.functional
from torch import Tensor

from .__pow import _pow


def pow_power_series(input: Tensor, exponent, maximum_exponent=None) -> Tensor:
    return _pow(torchaudio.functional.convolve, input, exponent, maximum_exponent)
