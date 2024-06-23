from .__vander_nd_flat import _vander_nd_flat
from ._legvander import legvander_vandermonde_1d


def legvander2d(x, y, deg):
    return _vander_nd_flat(
        (legvander_vandermonde_1d, legvander_vandermonde_1d), (x, y), deg
    )
