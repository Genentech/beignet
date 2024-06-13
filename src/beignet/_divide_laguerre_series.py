from beignet._multiply_laguerre_series import multiply_laguerre_series

from .polynomial.__div import _div


def divide_laguerre_series(c1, c2):
    return _div(multiply_laguerre_series, c1, c2)
