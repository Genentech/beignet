from beignet.polynomial import _as_series, legadd, legmulx


def poly2leg(pol):
    [pol] = _as_series([pol])
    deg = len(pol) - 1
    res = 0
    for i in range(deg, -1, -1):
        res = legadd(legmulx(res), pol[i])
    return res
