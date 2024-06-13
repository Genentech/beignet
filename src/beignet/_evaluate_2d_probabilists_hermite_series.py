from ._evaluate_probabilists_hermite_series import evaluate_probabilists_hermite_series
from .polynomial.__valnd import _valnd


def evaluate_2d_probabilists_hermite_series(x, y, c):
    return _valnd(evaluate_probabilists_hermite_series, c, x, y)
