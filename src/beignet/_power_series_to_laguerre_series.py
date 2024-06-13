from beignet._add_laguerre_series import add_laguerre_series

from .polynomial._as_series import as_series
from .polynomial._lagmulx import lagmulx


def power_series_to_laguerre_series(pol):
    [pol] = as_series([pol])
    res = 0
    for p in pol[::-1]:
        res = add_laguerre_series(lagmulx(res), p)
    return res
