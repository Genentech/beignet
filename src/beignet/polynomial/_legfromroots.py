from .__from_roots import _from_roots
from ._legendre_series_line import legendre_series_line
from ._multiply_legendre_series import multiply_legendre_series


def legfromroots(roots):
    return _from_roots(legendre_series_line, multiply_legendre_series, roots)
