from .__evaluate import _evaluate
from ._evaluate_1d_power_series import evaluate_1d_power_series


def evaluate_3d_power_series(x, y, z, c):
    return _evaluate(evaluate_1d_power_series, c, x, y, z)
