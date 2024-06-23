from .__fit import _fit
from ._hermvander import physicists_hermite_series_hermvander


def fit_physicists_hermite_series(x, y, deg, rcond=None, full=False, w=None):
    return _fit(physicists_hermite_series_hermvander, x, y, deg, rcond, full, w)
