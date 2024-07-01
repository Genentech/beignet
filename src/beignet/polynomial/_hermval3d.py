from torch import Tensor

from beignet.polynomial import _evaluate, hermval


def hermval3d(
    x: Tensor,
    y: Tensor,
    z: Tensor,
    c: Tensor,
) -> Tensor:
    return _evaluate(hermval, c, x, y, z)
