from ._as_series import as_series
from ._hermadd import hermadd
from ._hermmulx import hermmulx


def poly2herm(pol):
    [pol] = as_series([pol])
    deg = len(pol) - 1
    res = 0
    for i in range(deg, -1, -1):
        res = hermadd(hermmulx(res), pol[i])
    return res
