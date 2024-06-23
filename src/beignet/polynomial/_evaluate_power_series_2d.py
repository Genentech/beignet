from .__evaluate import _evaluate
from ._evaluate_power_series_1d import evaluate_power_series_1d


def evaluate_power_series_2d(x, y, c):
    return _evaluate(evaluate_power_series_1d, c, x, y)
