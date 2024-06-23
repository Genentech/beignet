from torch import Tensor

from .__evaluate import _evaluate
from ._evaluate_legendre_series_1d import evaluate_legendre_series_1d


def evaluate_legendre_series_2d(
    x: Tensor,
    y: Tensor,
    c: Tensor,
) -> Tensor:
    return _evaluate(
        evaluate_legendre_series_1d,
        c,
        x,
        y,
    )
