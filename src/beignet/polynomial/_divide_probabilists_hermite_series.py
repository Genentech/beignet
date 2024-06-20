from .__div import _div
from ._multiply_probabilists_hermite_series import multiply_probabilists_hermite_series


def divide_probabilists_hermite_series(c1, c2):
    return _div(multiply_probabilists_hermite_series, c1, c2)
