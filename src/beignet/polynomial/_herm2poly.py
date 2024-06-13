from ._add_power_series import add_power_series
from ._as_series import as_series
from ._polymulx import polymulx
from ._polysub import polysub


def herm2poly(c):
    [c] = as_series([c])
    n = len(c)
    if n == 1:
        return c
    if n == 2:
        c[1] *= 2
        return c
    else:
        c0 = c[-2]
        c1 = c[-1]

        for i in range(n - 1, 1, -1):
            tmp = c0
            c0 = polysub(c[i - 2], c1 * (2 * (i - 1)))
            c1 = add_power_series(tmp, polymulx(c1) * 2)
        return add_power_series(c0, polymulx(c1) * 2)
