from .__from_roots import _from_roots
from ._legline import legline
from ._multiply_legendre_series import multiply_legendre_series


def legfromroots(roots):
    return _from_roots(legline, multiply_legendre_series, roots)
