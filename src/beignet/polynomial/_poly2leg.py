from ._as_series import as_series
from ._legadd import legadd
from ._legmulx import legmulx


def poly2leg(pol):
    [pol] = as_series([pol])
    deg = len(pol) - 1
    res = 0
    for i in range(deg, -1, -1):
        res = legadd(legmulx(res), pol[i])
    return res
