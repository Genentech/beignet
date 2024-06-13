from beignet._add_chebyshev_series import add_chebyshev_series

from .polynomial._as_series import as_series
from .polynomial._chebmulx import chebmulx


def power_series_to_chebyshev_series(pol):
    [pol] = as_series([pol])
    deg = len(pol) - 1
    res = 0
    for i in range(deg, -1, -1):
        res = add_chebyshev_series(chebmulx(res), pol[i])
    return res
