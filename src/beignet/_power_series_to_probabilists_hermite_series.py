from beignet._add_probabilists_hermite_series import add_probabilists_hermite_series

from .polynomial._as_series import as_series
from .polynomial._hermemulx import hermemulx


def power_series_to_probabilists_hermite_series(pol):
    [pol] = as_series([pol])
    deg = len(pol) - 1
    res = 0
    for i in range(deg, -1, -1):
        res = add_probabilists_hermite_series(hermemulx(res), pol[i])
    return res
