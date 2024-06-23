from .__vander_nd_flat import _vander_nd_flat
from ._hermvander import physicists_hermite_series_hermvander


def physicists_hermite_series_hermvander2d(x, y, deg):
    return _vander_nd_flat(
        (physicists_hermite_series_hermvander, physicists_hermite_series_hermvander),
        (x, y),
        deg,
    )
