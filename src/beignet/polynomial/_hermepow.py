from beignet._multiply_probabilists_hermite_series import (
    multiply_probabilists_hermite_series,
)

from .__pow import _pow


def hermepow(c, pow, maxpower=16):
    return _pow(multiply_probabilists_hermite_series, c, pow, maxpower)
