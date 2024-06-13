from ._multiply_chebyshev_series import multiply_chebyshev_series
from .polynomial import _fromroots, chebline


def chebyshev_series_from_roots(roots):
    return _fromroots(chebline, multiply_chebyshev_series, roots)
