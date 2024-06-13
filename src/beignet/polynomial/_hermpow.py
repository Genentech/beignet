from beignet._multiply_physicists_hermite_series import (
    multiply_physicists_hermite_series,
)

from .__pow import _pow


def hermpow(c, pow, maxpower=16):
    return _pow(multiply_physicists_hermite_series, c, pow, maxpower)
