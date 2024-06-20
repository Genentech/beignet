from .__as_series import _as_series
from ._add_power_series import add_power_series
from ._polymulx import polymulx
from ._subtract_power_series import subtract_power_series


def legendre_series_to_power_series(c):
    [c] = _as_series([c])
    n = len(c)
    if n < 3:
        return c
    else:
        c0 = c[-2]
        c1 = c[-1]

        for i in range(n - 1, 1, -1):
            tmp = c0
            c0 = subtract_power_series(c[i - 2], (c1 * (i - 1)) / i)
            c1 = add_power_series(tmp, (polymulx(c1) * (2 * i - 1)) / i)
        return add_power_series(c0, polymulx(c1))
