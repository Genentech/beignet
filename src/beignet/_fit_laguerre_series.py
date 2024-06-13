from .polynomial.__fit import _fit
from .polynomial._lagvander import lagvander


def fit_laguerre_series(x, y, deg, rcond=None, full=False, w=None):
    return _fit(lagvander, x, y, deg, rcond, full, w)
