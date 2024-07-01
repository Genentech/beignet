from torch import Tensor

from beignet.polynomial import hermeval
from beignet.polynomial.__evaluate import _evaluate


def hermeval2d(
    x: Tensor,
    y: Tensor,
    c: Tensor,
) -> Tensor:
    return _evaluate(hermeval, c, x, y)
