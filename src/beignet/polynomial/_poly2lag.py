from ._add_laguerre_series import add_laguerre_series
from ._as_series import as_series
from ._lagmulx import lagmulx


def poly2lag(pol):
    [pol] = as_series([pol])
    res = 0
    for p in pol[::-1]:
        res = add_laguerre_series(lagmulx(res), p)
    return res
