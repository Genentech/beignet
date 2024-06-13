from ._add_chebyshev_series import add_chebyshev_polynomial
from ._as_series import as_series
from ._chebmulx import chebmulx


def poly2cheb(pol):
    [pol] = as_series([pol])
    deg = len(pol) - 1
    res = 0
    for i in range(deg, -1, -1):
        res = add_chebyshev_polynomial(chebmulx(res), pol[i])
    return res
