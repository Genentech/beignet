from .__pow import _pow
from ._multiply_probabilists_hermite_series import multiply_probabilists_hermite_series


def hermepow(c, pow, maxpower=16):
    return _pow(multiply_probabilists_hermite_series, c, pow, maxpower)
