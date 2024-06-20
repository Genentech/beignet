import torch

from .__as_series import _as_series
from ._hermadd import hermadd
from ._hermmulx import hermmulx
from ._hermsub import hermsub


def hermmul(input, other):
    input, other = _as_series([input, other])

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
            c0 = hermsub(c[-i] * xs, input * (2 * (nd - 1)))
            input = hermadd(tmp, hermmulx(input) * 2)

    input = torch.tensor(input)
    input = torch.ravel(input)

    output = hermmulx(input) * 2

    output = hermadd(c0, output)

    return output
