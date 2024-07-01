from torch import Tensor

from beignet.polynomial import chebval
from beignet.polynomial.__evaluate import _evaluate


def chebval2d(
    x: Tensor,
    y: Tensor,
    c: Tensor,
) -> Tensor:
    return _evaluate(chebval, c, x, y)
