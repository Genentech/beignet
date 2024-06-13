from beignet._add_power_series import add_power_series
from beignet._subtract_power_series import subtract_power_series

from .polynomial import polymulx
from .polynomial._as_series import as_series


def leg_to_power_series(c):
    [c] = as_series([c])
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
