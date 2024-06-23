from .__fit import _fit
from ._hermevander import probabilists_hermite_series_hermevander


def fit_probabilists_hermite_series(x, y, deg, rcond=None, full=False, w=None):
    return _fit(probabilists_hermite_series_hermevander, x, y, deg, rcond, full, w)
