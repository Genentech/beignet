from beignet.polynomial import _as_series, lagadd, lagmulx


def poly2lag(pol):
    [pol] = _as_series([pol])
    res = 0
    for p in pol[::-1]:
        res = lagadd(lagmulx(res), p)
    return res
