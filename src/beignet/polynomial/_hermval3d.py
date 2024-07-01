from torch import Tensor

from beignet.polynomial import hermval
from beignet.polynomial.__evaluate import _evaluate


def hermval3d(
    x: Tensor,
    y: Tensor,
    z: Tensor,
    c: Tensor,
) -> Tensor:
    return _evaluate(hermval, c, x, y, z)
