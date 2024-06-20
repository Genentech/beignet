from .__evaluate import _evaluate
from ._evaluate_1d_chebyshev_series import evaluate_1d_chebyshev_series


def evaluate_2d_chebyshev_series(x, y, c):
    return _evaluate(evaluate_1d_chebyshev_series, c, x, y)
