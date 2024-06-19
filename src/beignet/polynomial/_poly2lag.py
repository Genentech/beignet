from .__as_series import _as_series
from ._lagadd import lagadd
from ._lagmulx import lagmulx


def poly2lag(pol):
    [pol] = _as_series([pol])
    res = 0
    for p in pol[::-1]:
        res = lagadd(lagmulx(res), p)
    return res
