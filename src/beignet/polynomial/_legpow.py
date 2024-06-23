from .__pow import _pow
from ._multiply_legendre_series import multiply_legendre_series


def pow_legendre_series(c, pow, maxpower=16):
    return _pow(multiply_legendre_series, c, pow, maxpower)
