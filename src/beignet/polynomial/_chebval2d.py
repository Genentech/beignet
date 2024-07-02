from torch import Tensor

from .__evaluate import _evaluate
from ._chebval import chebval


def chebval2d(
    x: Tensor,
    y: Tensor,
    c: Tensor,
) -> Tensor:
    return _evaluate(chebval, c, x, y)
