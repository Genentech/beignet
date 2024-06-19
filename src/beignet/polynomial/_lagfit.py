from .__fit import _fit
from ._lagvander import lagvander


def lagfit(x, y, deg, rcond=None, full=False, w=None):
    return _fit(lagvander, x, y, deg, rcond, full, w)
