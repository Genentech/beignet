from .__fit import _fit
from ._legendre_series_vandermonde_1d import legendre_series_vandermonde_1d


def legfit(x, y, deg, rcond=None, full=False, w=None):
    return _fit(legendre_series_vandermonde_1d, x, y, deg, rcond, full, w)
