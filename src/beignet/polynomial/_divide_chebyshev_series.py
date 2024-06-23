from torch import Tensor

from .__as_series import _as_series
from .__c_series_to_z_series import _c_series_to_z_series
from .__trim_sequence import _trim_sequence
from .__z_series_div import _z_series_div
from .__z_series_to_c_series import _z_series_to_c_series


def divide_chebyshev_series(
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
        z1 = _c_series_to_z_series(input)
        z2 = _c_series_to_z_series(other)

        quo, rem = _z_series_div(z1, z2)

        quo = _trim_sequence(_z_series_to_c_series(quo))
        rem = _trim_sequence(_z_series_to_c_series(rem))

        return quo, rem
