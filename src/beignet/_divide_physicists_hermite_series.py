from beignet._multiply_physicists_hermite_series import (
    multiply_physicists_hermite_series,
)

from .polynomial.__div import _div


def divide_physicists_hermite_series(c1, c2):
    return _div(multiply_physicists_hermite_series, c1, c2)
