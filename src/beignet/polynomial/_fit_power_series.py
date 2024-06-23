from .__fit import _fit
from ._power_series_vandermonde_1d import power_series_vandermonde_1d


def fit_power_series(x, y, deg, rcond=None, full=False, w=None):
    return _fit(power_series_vandermonde_1d, x, y, deg, rcond, full, w)
