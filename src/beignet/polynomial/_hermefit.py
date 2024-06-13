from .__fit import _fit
from ._hermevander import hermevander


def hermefit(x, y, deg, rcond=None, full=False, w=None):
    return _fit(hermevander, x, y, deg, rcond, full, w)
