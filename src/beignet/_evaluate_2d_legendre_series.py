from ._evaluate_legendre_series import evaluate_legendre_series
from .polynomial.__valnd import _valnd


def evaluate_2d_legendre_series(x, y, c):
    return _valnd(evaluate_legendre_series, c, x, y)
