from torch import Tensor

from .__evaluate import _evaluate
from ._hermval import hermval


def hermval2d(
    x: Tensor,
    y: Tensor,
    c: Tensor,
) -> Tensor:
    return _evaluate(hermval, c, x, y)
