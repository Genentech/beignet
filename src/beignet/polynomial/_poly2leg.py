from ._add_legendre_series import add_legendre_series
from ._as_series import as_series
from ._legmulx import legmulx


def poly2leg(pol):
    [pol] = as_series([pol])
    deg = len(pol) - 1
    res = 0
    for i in range(deg, -1, -1):
        res = add_legendre_series(legmulx(res), pol[i])
    return res
