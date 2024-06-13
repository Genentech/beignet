from beignet._add_power_series import add_power_series
from beignet._subtract_power_series import subtract_power_series

from .polynomial import polymulx
from .polynomial._as_series import as_series


def lag_to_power_series(c):
    [c] = as_series([c])
    n = len(c)
    if n == 1:
        return c
    else:
        c0 = c[-2]
        c1 = c[-1]

        for i in range(n - 1, 1, -1):
            tmp = c0
            c0 = subtract_power_series(c[i - 2], (c1 * (i - 1)) / i)
            c1 = add_power_series(
                tmp, subtract_power_series((2 * i - 1) * c1, polymulx(c1)) / i
            )
        return add_power_series(c0, subtract_power_series(c1, polymulx(c1)))
