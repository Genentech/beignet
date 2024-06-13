from ._evaluate_physicists_hermite_series import evaluate_physicists_hermite_series
from .polynomial.__valnd import _valnd


def evaluate_2d_physicists_hermite_series(x, y, c):
    return _valnd(evaluate_physicists_hermite_series, c, x, y)
