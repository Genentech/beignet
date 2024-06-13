from .polynomial.__fit import _fit
from .polynomial._hermvander import hermvander


def fit_physicists_hermite_series(x, y, deg, rcond=None, full=False, w=None):
    return _fit(hermvander, x, y, deg, rcond, full, w)
