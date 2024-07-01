from torch import Tensor

from beignet.polynomial import _evaluate, lagval


def lagval2d(
    x: Tensor,
    y: Tensor,
    c: Tensor,
) -> Tensor:
    return _evaluate(lagval, c, x, y)
