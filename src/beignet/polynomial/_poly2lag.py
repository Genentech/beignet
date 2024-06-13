from ._as_series import as_series
from ._lagadd import lagadd
from ._lagmulx import lagmulx


def poly2lag(pol):
    [pol] = as_series([pol])
    res = 0
    for p in pol[::-1]:
        res = lagadd(lagmulx(res), p)
    return res
