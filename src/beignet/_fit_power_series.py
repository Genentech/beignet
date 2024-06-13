from .polynomial.__fit import _fit
from .polynomial._polyvander import polyvander


def fit_power_series(x, y, deg, rcond=None, full=False, w=None):
    return _fit(polyvander, x, y, deg, rcond, full, w)
