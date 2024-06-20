from .__div import _div
from ._multiply_legendre_series import multiply_legendre_series


def divide_legendre_series(c1, c2):
    return _div(multiply_legendre_series, c1, c2)
