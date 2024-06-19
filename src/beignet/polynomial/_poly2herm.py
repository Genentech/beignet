from .__as_series import _as_series
from ._hermadd import hermadd
from ._hermmulx import hermmulx


def poly2herm(input):
    [input] = _as_series([input])
    deg = len(input) - 1
    res = 0
    for i in range(deg, -1, -1):
        res = hermadd(hermmulx(res), input[i])
    return res
