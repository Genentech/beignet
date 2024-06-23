from .__fit import _fit
from ._laguerre_series_vandermonde_1d import laguerre_series_vandermonde_1d


def fit_laguerre_series(x, y, deg, rcond=None, full=False, w=None):
    return _fit(laguerre_series_vandermonde_1d, x, y, deg, rcond, full, w)
