from torch import Tensor

from beignet.polynomial import lagval
from beignet.polynomial.__evaluate import _evaluate


def lagval3d(
    x: Tensor,
    y: Tensor,
    z: Tensor,
    c: Tensor,
) -> Tensor:
    return _evaluate(lagval, c, x, y, z)
