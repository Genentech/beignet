from .__vander_nd_flat import _vander_nd_flat
from ._physicists_hermite_series_vandermonde_1d import (
    physicists_hermite_series_vandermonde_1d,
)


def physicists_hermite_series_vandermonde_3d(x, y, z, deg):
    return _vander_nd_flat(
        (
            physicists_hermite_series_vandermonde_1d,
            physicists_hermite_series_vandermonde_1d,
            physicists_hermite_series_vandermonde_1d,
        ),
        (x, y, z),
        deg,
    )
