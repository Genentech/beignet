from beignet._multiply_probabilists_hermite_series import (
    multiply_probabilists_hermite_series,
)

from .polynomial.__div import _div


def divide_probabilists_hermite_series(c1, c2):
    return _div(multiply_probabilists_hermite_series, c1, c2)
