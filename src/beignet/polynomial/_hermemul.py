from .__as_series import _as_series
from ._hermeadd import hermeadd
from ._hermemulx import hermemulx
from ._hermesub import hermesub


def hermemul(input, other):
    (
        input,
        other,
    ) = _as_series([input, other])

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
            c0 = hermesub(c[-i] * xs, input * (nd - 1))
            input = hermeadd(tmp, hermemulx(input))

    output = hermemulx(input)

    output = hermeadd(c0, output)

    return output
