from beignet._chebyshev_series_vandermonde import chebyshev_series_vandermonde

from .__vander_nd_flat import _vander_nd_flat


def chebvander2d(x, y, deg):
    return _vander_nd_flat(
        (chebyshev_series_vandermonde, chebyshev_series_vandermonde), (x, y), deg
    )
