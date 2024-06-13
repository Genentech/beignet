from beignet._legendre_series_vandermonde import legendre_series_vandermonde

from .__vander_nd_flat import _vander_nd_flat


def legvander3d(x, y, z, deg):
    return _vander_nd_flat(
        (
            legendre_series_vandermonde,
            legendre_series_vandermonde,
            legendre_series_vandermonde,
        ),
        (x, y, z),
        deg,
    )
