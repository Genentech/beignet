from .__from_roots import _from_roots
from ._multiply_power_series import multiply_power_series
from ._power_series_line import power_series_line


def polyfromroots(roots):
    return _from_roots(power_series_line, multiply_power_series, roots)
