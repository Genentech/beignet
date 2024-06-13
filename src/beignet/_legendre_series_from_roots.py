from beignet._multiply_legendre_series import multiply_legendre_series

from .polynomial.__fromroots import _fromroots
from .polynomial._legline import legline


def legendre_series_from_roots(roots):
    return _fromroots(legline, multiply_legendre_series, roots)
