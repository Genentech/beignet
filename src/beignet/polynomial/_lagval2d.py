from torch import Tensor

from beignet.polynomial import lagval
from beignet.polynomial.__evaluate import _evaluate


def lagval2d(
    x: Tensor,
    y: Tensor,
    c: Tensor,
) -> Tensor:
    return _evaluate(lagval, c, x, y)
