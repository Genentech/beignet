from torch import Tensor

from .__vander_nd_flat import _vander_nd_flat
from ._power_series_vandermonde_1d import power_series_vandermonde_1d


def power_series_vandermonde_3d(x: Tensor, y: Tensor, z: Tensor, deg):
    return _vander_nd_flat(
        (
            power_series_vandermonde_1d,
            power_series_vandermonde_1d,
            power_series_vandermonde_1d,
        ),
        (x, y, z),
        deg,
    )
