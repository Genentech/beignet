from .__fit import _fit
from ._legvander import legvander


def legfit(x, y, deg, rcond=None, full=False, w=None):
    return _fit(legvander, x, y, deg, rcond, full, w)
