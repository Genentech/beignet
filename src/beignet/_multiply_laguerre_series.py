from beignet._add_laguerre_series import add_laguerre_series
from beignet._subtract_laguerre_series import subtract_laguerre_series

from .polynomial._as_series import as_series
from .polynomial._lagmulx import lagmulx


def multiply_laguerre_series(c1, c2):
    [c1, c2] = as_series([c1, c2])

    if len(c1) > len(c2):
        c = c2
        xs = c1
    else:
        c = c1
        xs = c2

    if len(c) == 1:
        c0 = c[0] * xs
        c1 = 0
    elif len(c) == 2:
        c0 = c[0] * xs
        c1 = c[1] * xs
    else:
        nd = len(c)
        c0 = c[-2] * xs
        c1 = c[-1] * xs
        for i in range(3, len(c) + 1):
            tmp = c0
            nd = nd - 1
            c0 = subtract_laguerre_series(c[-i] * xs, (c1 * (nd - 1)) / nd)
            c1 = add_laguerre_series(
                tmp, subtract_laguerre_series((2 * nd - 1) * c1, lagmulx(c1)) / nd
            )
    return add_laguerre_series(c0, subtract_laguerre_series(c1, lagmulx(c1)))
