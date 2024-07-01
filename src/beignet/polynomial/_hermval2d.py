from torch import Tensor

from beignet.polynomial import hermval
from beignet.polynomial.__evaluate import _evaluate


def hermval2d(
    x: Tensor,
    y: Tensor,
    c: Tensor,
) -> Tensor:
    return _evaluate(hermval, c, x, y)
