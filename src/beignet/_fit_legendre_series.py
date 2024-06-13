from beignet._legendre_series_vandermonde import legendre_series_vandermonde

from .polynomial.__fit import _fit


def fit_legendre_series(x, y, deg, rcond=None, full=False, w=None):
    return _fit(legendre_series_vandermonde, x, y, deg, rcond, full, w)
