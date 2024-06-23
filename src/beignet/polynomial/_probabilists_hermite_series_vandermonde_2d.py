from .__vander_nd_flat import _vander_nd_flat
from ._probabilists_hermite_series_vandermonde_1d import (
    probabilists_hermite_series_vandermonde_1d,
)


def probabilists_hermite_series_vandermonde_2d(x, y, deg):
    return _vander_nd_flat(
        (
            probabilists_hermite_series_vandermonde_1d,
            probabilists_hermite_series_vandermonde_1d,
        ),
        (x, y),
        deg,
    )
