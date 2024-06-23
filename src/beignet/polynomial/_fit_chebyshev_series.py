from .__fit import _fit
from ._chebyshev_series_vandermonde_1d import chebyshev_series_vandermonde_1d


def fit_chebyshev_series(x, y, deg, rcond=None, full=False, w=None):
    return _fit(chebyshev_series_vandermonde_1d, x, y, deg, rcond, full, w)
