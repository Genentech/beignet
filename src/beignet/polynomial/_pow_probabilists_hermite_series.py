from .__pow import _pow
from ._multiply_probabilists_hermite_series import multiply_probabilists_hermite_series


def pow_probabilists_hermite_series(c, pow, maxpower=16):
    return _pow(multiply_probabilists_hermite_series, c, pow, maxpower)
