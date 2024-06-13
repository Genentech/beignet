from ._evaluate_power_series import evaluate_power_series
from .polynomial.__valnd import _valnd


def evaluate_3d_power_series(x, y, z, c):
    return _valnd(evaluate_power_series, c, x, y, z)
