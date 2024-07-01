from torch import Tensor

from beignet.polynomial import hermeval
from beignet.polynomial.__evaluate import _evaluate


def hermeval3d(
    x: Tensor,
    y: Tensor,
    z: Tensor,
    c: Tensor,
) -> Tensor:
    return _evaluate(hermeval, c, x, y, z)
