from .__div import _div
from ._multiply_laguerre_series import multiply_laguerre_series


def divide_laguerre_series(c1, c2):
    return _div(multiply_laguerre_series, c1, c2)
