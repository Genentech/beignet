from .__fit import _fit
from ._hermvander import hermvander


def hermfit(x, y, deg, rcond=None, full=False, w=None):
    return _fit(hermvander, x, y, deg, rcond, full, w)
