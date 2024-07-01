from torch import Tensor

from beignet.polynomial import _evaluate, hermeval


def hermeval2d(
    x: Tensor,
    y: Tensor,
    c: Tensor,
) -> Tensor:
    return _evaluate(hermeval, c, x, y)
