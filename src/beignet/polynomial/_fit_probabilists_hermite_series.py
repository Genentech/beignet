from .__fit import _fit
from ._probabilists_hermite_series_vandermonde_1d import (
    probabilists_hermite_series_vandermonde_1d,
)


def fit_probabilists_hermite_series(x, y, deg, rcond=None, full=False, w=None):
    return _fit(probabilists_hermite_series_vandermonde_1d, x, y, deg, rcond, full, w)
