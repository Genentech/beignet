from beignet.polynomial import _as_series, hermeadd, hermemulx


def poly2herme(input):
    [input] = _as_series([input])
    deg = len(input) - 1
    res = 0
    for i in range(deg, -1, -1):
        res = hermeadd(hermemulx(res), input[i])
    return res
