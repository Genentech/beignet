from beignet._evaluate_power_series import evaluate_power_series

from .__gridnd import _gridnd


def polygrid2d(x, y, c):
    return _gridnd(evaluate_power_series, c, x, y)
