from .__vander_nd_flat import _vander_nd_flat
from ._hermevander import probabilists_hermite_series_hermevander


def probabilists_hermite_series_hermevander2d(x, y, deg):
    return _vander_nd_flat(
        (
            probabilists_hermite_series_hermevander,
            probabilists_hermite_series_hermevander,
        ),
        (x, y),
        deg,
    )
