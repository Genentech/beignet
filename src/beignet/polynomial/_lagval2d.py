from torch import Tensor

from .__evaluate import _evaluate
from ._lagval import lagval


def lagval2d(
    x: Tensor,
    y: Tensor,
    c: Tensor,
) -> Tensor:
    return _evaluate(lagval, c, x, y)
