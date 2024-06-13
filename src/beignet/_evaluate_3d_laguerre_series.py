from ._evaluate_laguerre_series import evaluate_laguerre_series
from .polynomial.__valnd import _valnd


def evaluate_3d_laguerre_series(x, y, z, c):
    return _valnd(evaluate_laguerre_series, c, x, y, z)
