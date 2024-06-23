from .__vander_nd_flat import _vander_nd_flat
from ._chebyshev_series_vandermonde_1d import chebyshev_series_vandermonde_1d


def chebyshev_series_vandermonde_3d(x, y, z, degree):
    return _vander_nd_flat(
        (
            chebyshev_series_vandermonde_1d,
            chebyshev_series_vandermonde_1d,
            chebyshev_series_vandermonde_1d,
        ),
        (x, y, z),
        degree,
    )
