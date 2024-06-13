from ._add_physicists_hermite_series import add_physicists_hermite_series
from ._as_series import as_series
from ._hermmulx import hermmulx


def poly2herm(pol):
    [pol] = as_series([pol])
    deg = len(pol) - 1
    res = 0
    for i in range(deg, -1, -1):
        res = add_physicists_hermite_series(hermmulx(res), pol[i])
    return res
