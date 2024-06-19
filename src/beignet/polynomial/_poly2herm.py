from beignet.polynomial import _as_series, hermadd, hermmulx


def poly2herm(input):
    [input] = _as_series([input])
    deg = len(input) - 1
    res = 0
    for i in range(deg, -1, -1):
        res = hermadd(hermmulx(res), input[i])
    return res
