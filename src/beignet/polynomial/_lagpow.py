from .__pow import _pow
from ._multiply_laguerre_series import multiply_laguerre_series


def pow_laguerre_series(c, pow, maxpower=16):
    return _pow(multiply_laguerre_series, c, pow, maxpower)
