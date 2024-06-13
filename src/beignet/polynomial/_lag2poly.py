from ._add_power_series import add_power_series
from ._as_series import as_series
from ._polymulx import polymulx
from ._polysub import polysub


def lag2poly(c):
    [c] = as_series([c])
    n = len(c)
    if n == 1:
        return c
    else:
        c0 = c[-2]
        c1 = c[-1]

        for i in range(n - 1, 1, -1):
            tmp = c0
            c0 = polysub(c[i - 2], (c1 * (i - 1)) / i)
            c1 = add_power_series(tmp, polysub((2 * i - 1) * c1, polymulx(c1)) / i)
        return add_power_series(c0, polysub(c1, polymulx(c1)))
