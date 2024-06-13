from beignet._evaluate_chebyshev_series import evaluate_chebyshev_series

from .__gridnd import _gridnd


def chebgrid2d(x, y, c):
    return _gridnd(evaluate_chebyshev_series, c, x, y)
