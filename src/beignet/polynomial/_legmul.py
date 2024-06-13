from ._as_series import as_series
from ._legadd import legadd
from ._legmulx import legmulx
from ._legsub import legsub


def legmul(c1, c2):
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
            c0 = legsub(c[-i] * xs, (c1 * (nd - 1)) / nd)
            c1 = legadd(tmp, (legmulx(c1) * (2 * nd - 1)) / nd)
    return legadd(c0, legmulx(c1))
