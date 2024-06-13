from beignet._multiply_legendre_series import multiply_legendre_series

from .__pow import _pow


def legpow(c, pow, maxpower=16):
    return _pow(multiply_legendre_series, c, pow, maxpower)
