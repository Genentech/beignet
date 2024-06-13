from beignet._chebyshev_series_vandermonde import chebyshev_series_vandermonde

from .polynomial.__fit import _fit


def fit_chebyshev_series(x, y, deg, rcond=None, full=False, w=None):
    return _fit(chebyshev_series_vandermonde, x, y, deg, rcond, full, w)
