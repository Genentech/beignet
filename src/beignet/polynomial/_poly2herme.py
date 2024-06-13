from ._add_probabilists_hermite_series import add_probabilists_hermite_series
from ._as_series import as_series
from ._hermemulx import hermemulx


def poly2herme(pol):
    [pol] = as_series([pol])
    deg = len(pol) - 1
    res = 0
    for i in range(deg, -1, -1):
        res = add_probabilists_hermite_series(hermemulx(res), pol[i])
    return res
