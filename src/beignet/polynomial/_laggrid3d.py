from beignet._evaluate_laguerre_series import evaluate_laguerre_series

from .__gridnd import _gridnd


def laggrid3d(x, y, z, c):
    return _gridnd(evaluate_laguerre_series, c, x, y, z)
