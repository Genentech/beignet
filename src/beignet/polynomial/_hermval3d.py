from torch import Tensor

from .__evaluate import _evaluate
from ._hermval import hermval


def hermval3d(
    x: Tensor,
    y: Tensor,
    z: Tensor,
    c: Tensor,
) -> Tensor:
    return _evaluate(hermval, c, x, y, z)
