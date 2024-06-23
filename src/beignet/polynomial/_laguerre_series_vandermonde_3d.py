from .__vander_nd_flat import _vander_nd_flat
from ._laguerre_series_vandermonde_1d import laguerre_series_vandermonde_1d


def laguerre_series_vandermonde_3d(x, y, z, deg):
    return _vander_nd_flat(
        (
            laguerre_series_vandermonde_1d,
            laguerre_series_vandermonde_1d,
            laguerre_series_vandermonde_1d,
        ),
        (x, y, z),
        deg,
    )
