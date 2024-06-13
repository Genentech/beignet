from ._evaluate_power_series import evaluate_power_series
from .polynomial.__valnd import _valnd


def evaluate_2d_power_series(x, y, c):
    return _valnd(evaluate_power_series, c, x, y)
