from .__as_series import _as_series
from ._add_power_series import add_power_series
from ._polymulx import polymulx
from ._subtract_power_series import subtract_power_series


def chebyshev_series_to_power_series(input):
    [input] = _as_series([input])

    n = len(input)
    if n < 3:
        return input
    else:
        c0 = input[-2]
        c1 = input[-1]

        for i in range(n - 1, 1, -1):
            tmp = c0
            c0 = subtract_power_series(input[i - 2], c1)
            c1 = add_power_series(tmp, polymulx(c1) * 2)
        return add_power_series(c0, polymulx(c1))
