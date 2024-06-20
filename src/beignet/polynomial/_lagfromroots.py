from .__from_roots import _from_roots
from ._lagline import lagline
from ._multiply_laguerre_series import multiply_laguerre_series


def lagfromroots(roots):
    return _from_roots(lagline, multiply_laguerre_series, roots)
