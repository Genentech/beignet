from .__div import _div
from ._multiply_physicists_hermite_series import multiply_physicists_hermite_series


def divide_physicists_hermite_series(c1, c2):
    return _div(multiply_physicists_hermite_series, c1, c2)
