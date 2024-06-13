from beignet._multiply_laguerre_series import multiply_laguerre_series

from .polynomial.__fromroots import _fromroots
from .polynomial._lagline import lagline


def laguerre_series_from_roots(roots):
    return _fromroots(lagline, multiply_laguerre_series, roots)
