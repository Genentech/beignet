from beignet._multiply_laguerre_series import multiply_laguerre_series

from .__pow import _pow


def lagpow(c, pow, maxpower=16):
    return _pow(multiply_laguerre_series, c, pow, maxpower)
