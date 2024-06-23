from .__evaluate import _evaluate
from ._evaluate_legendre_series_1d import evaluate_legendre_series_1d


def evaluate_legendre_series_2d(x, y, c):
    return _evaluate(evaluate_legendre_series_1d, c, x, y)
