from .__from_roots import _from_roots
from ._lagline import laguerre_series_line
from ._multiply_laguerre_series import multiply_laguerre_series


def lagfromroots(roots):
    return _from_roots(laguerre_series_line, multiply_laguerre_series, roots)
