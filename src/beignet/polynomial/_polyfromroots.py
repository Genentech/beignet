from .__from_roots import _from_roots
from ._multiply_power_series import multiply_power_series
from ._polyline import polyline


def polyfromroots(roots):
    return _from_roots(polyline, multiply_power_series, roots)
