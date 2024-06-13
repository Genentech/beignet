from ._evaluate_laguerre_series import evaluate_laguerre_series
from .polynomial.__valnd import _valnd


def evaluate_2d_laguerre_series(x, y, c):
    return _valnd(evaluate_laguerre_series, c, x, y)
