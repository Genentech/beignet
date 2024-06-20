from .__evaluate import _evaluate
from ._evaluate_1d_legendre_series import evaluate_1d_legendre_series


def evaluate_2d_legendre_series(x, y, c):
    return _evaluate(evaluate_1d_legendre_series, c, x, y)
