from torch import Tensor

from beignet.polynomial import _evaluate, hermval


def hermval2d(
    x: Tensor,
    y: Tensor,
    c: Tensor,
) -> Tensor:
    return _evaluate(hermval, c, x, y)
