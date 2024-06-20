from .__evaluate import _evaluate
from ._evaluate_1d_laguerre_series import evaluate_1d_laguerre_series


def evaluate_3d_laguerre_series(x, y, z, c):
    return _evaluate(evaluate_1d_laguerre_series, c, x, y, z)
