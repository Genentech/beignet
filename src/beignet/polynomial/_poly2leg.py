from .__as_series import _as_series
from ._legadd import legadd
from ._legmulx import legmulx


def poly2leg(pol):
    [pol] = _as_series([pol])
    deg = len(pol) - 1
    res = 0
    for i in range(deg, -1, -1):
        res = legadd(legmulx(res), pol[i])
    return res
