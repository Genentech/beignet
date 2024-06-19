from .__as_series import _as_series
from ._legadd import legadd
from ._legmulx import legmulx
from ._legsub import legsub


def legmul(input, other):
    [input, other] = _as_series([input, other])

    if len(input) > len(other):
        c = other
        xs = input
    else:
        c = input
        xs = other

    if len(c) == 1:
        c0 = c[0] * xs
        input = 0
    elif len(c) == 2:
        c0 = c[0] * xs
        input = c[1] * xs
    else:
        nd = len(c)
        c0 = c[-2] * xs
        input = c[-1] * xs
        for i in range(3, len(c) + 1):
            tmp = c0
            nd = nd - 1
            c0 = legsub(c[-i] * xs, (input * (nd - 1)) / nd)
            input = legadd(tmp, (legmulx(input) * (2 * nd - 1)) / nd)

    return legadd(c0, legmulx(input))
