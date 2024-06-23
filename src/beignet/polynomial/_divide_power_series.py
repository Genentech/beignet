from torch import Tensor

from .__as_series import _as_series
from .__trim_sequence import _trim_sequence


def divide_power_series(
    input: Tensor,
    other: Tensor,
) -> (Tensor, Tensor):
    [input, other] = _as_series([input, other])
    if other[-1] == 0:
        raise ZeroDivisionError()

    lc1 = len(input)
    lc2 = len(other)
    if lc1 < lc2:
        return input[:1] * 0, input
    elif lc2 == 1:
        return input / other[-1], input[:1] * 0
    else:
        dlen = lc1 - lc2
        scl = other[-1]
        other = other[:-1] / scl
        i = dlen
        j = lc1 - 1
        while i >= 0:
            input[i:j] -= other * input[j]
            i -= 1
            j -= 1
        return input[j + 1 :] / scl, _trim_sequence(input[: j + 1])
