from torch import Tensor

from .__evaluate import _evaluate
from ._evaluate_probabilists_hermite_series_1d import (
    evaluate_probabilists_hermite_series_1d,
)


def evaluate_probabilists_hermite_series_2d(
    x: Tensor,
    y: Tensor,
    c: Tensor,
) -> Tensor:
    return _evaluate(
        evaluate_probabilists_hermite_series_1d,
        c,
        x,
        y,
    )
