from .polynomial.__fit import _fit
from .polynomial._hermevander import hermevander


def fit_probabilists_hermite_series(x, y, deg, rcond=None, full=False, w=None):
    return _fit(hermevander, x, y, deg, rcond, full, w)
