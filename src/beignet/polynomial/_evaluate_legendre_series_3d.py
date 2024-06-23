from .__evaluate import _evaluate
from ._evaluate_legendre_series_1d import evaluate_legendre_series_1d


def evaluate_legendre_series_3d(x, y, z, c):
    return _evaluate(evaluate_legendre_series_1d, c, x, y, z)
