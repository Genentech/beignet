from torch import Tensor

from .__evaluate import _evaluate
from ._hermeval import hermeval


def hermeval2d(
    x: Tensor,
    y: Tensor,
    c: Tensor,
) -> Tensor:
    return _evaluate(hermeval, c, x, y)
