from ._evaluate_chebyshev_series import evaluate_chebyshev_series
from .polynomial.__valnd import _valnd


def evaluate_2d_chebyshev_series(x, y, c):
    return _valnd(evaluate_chebyshev_series, c, x, y)
