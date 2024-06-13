from beignet._multiply_power_series import multiply_power_series

from .polynomial.__fromroots import _fromroots
from .polynomial._polyline import polyline


def power_series_from_roots(roots):
    return _fromroots(polyline, multiply_power_series, roots)
