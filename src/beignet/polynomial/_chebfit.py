from .__fit import _fit
from ._chebvander import chebvander


def chebfit(x, y, deg, rcond=None, full=False, w=None):
    return _fit(chebvander, x, y, deg, rcond, full, w)
