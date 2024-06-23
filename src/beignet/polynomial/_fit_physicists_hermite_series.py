from .__fit import _fit
from ._physicists_hermite_series_vandermonde_1d import (
    physicists_hermite_series_vandermonde_1d,
)


def fit_physicists_hermite_series(x, y, deg, rcond=None, full=False, w=None):
    return _fit(physicists_hermite_series_vandermonde_1d, x, y, deg, rcond, full, w)
