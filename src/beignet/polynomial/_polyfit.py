from .__fit import _fit
from ._polyvander import polyvander


def polyfit(x, y, deg, rcond=None, full=False, w=None):
    return _fit(polyvander, x, y, deg, rcond, full, w)
