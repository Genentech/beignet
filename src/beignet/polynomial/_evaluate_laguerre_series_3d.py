from torch import Tensor

from .__evaluate import _evaluate
from ._evaluate_laguerre_series_1d import evaluate_laguerre_series_1d


def evaluate_laguerre_series_3d(
    x: Tensor,
    y: Tensor,
    z: Tensor,
    c: Tensor,
) -> Tensor:
    return _evaluate(
        evaluate_laguerre_series_1d,
        c,
        x,
        y,
        z,
    )
