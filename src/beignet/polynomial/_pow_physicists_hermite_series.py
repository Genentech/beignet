from .__pow import _pow
from ._multiply_physicists_hermite_series import multiply_physicists_hermite_series


def pow_physicists_hermite_series(c, pow, maxpower=16):
    return _pow(multiply_physicists_hermite_series, c, pow, maxpower)
